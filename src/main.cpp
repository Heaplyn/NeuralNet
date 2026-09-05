#include "ringwrapper.hpp"
#include "ring0/cuda_backend.hpp"
#include "ring1/recursive_layer.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include "ring2/vocab_manager.hpp"
#include "ring3/data_loader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace std;

// Attempts to slurp a text corpus from disk. Tries each candidate path in order
// and returns the first one that opens successfully. Returns empty on failure.
static string load_corpus_from_disk(const vector<string> &candidate_paths)
{
    for (const auto &path : candidate_paths)
    {
        ifstream file(path, ios::binary);
        if (!file.is_open())
            continue;
        stringstream buffer;
        buffer << file.rdbuf();
        string contents = buffer.str();
        if (!contents.empty())
        {
            cout << "  Loaded corpus from: " << path
                 << " (" << contents.size() << " bytes)\n";
            return contents;
        }
    }
    return {};
}

// Scans multi-file code dataset directory (e.g. data/code_dataset/) for all .cpp/.hpp files
[[maybe_unused]] static string load_code_corpus_multi()
{
    string combined;
    vector<string> search_dirs = {"data/code_dataset", "../data/code_dataset", "../../data/code_dataset"};
    for (const auto &dir_path : search_dirs)
    {
        if (filesystem::exists(dir_path) && filesystem::is_directory(dir_path))
        {
            size_t file_count = 0;
            for (const auto &entry : filesystem::recursive_directory_iterator(dir_path))
            {
                if (entry.is_regular_file() && (entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp"))
                {
                    ifstream f(entry.path());
                    if (f.is_open())
                    {
                        stringstream buf;
                        buf << f.rdbuf();
                        combined += "\n\n// File: " + entry.path().filename().string() + "\n";
                        combined += buf.str();
                        file_count++;
                    }
                }
            }
            if (file_count > 0)
            {
                cout << "  Loaded " << file_count << " C++ source files from " << dir_path << " (" << combined.size() << " bytes)\n";
                return combined;
            }
        }
    }
    return "";
}

// Parses CSV code dataset (e.g. data/code_dataset.csv with id,prompt,response)
[[maybe_unused]] static string load_code_from_csv(const vector<string> &candidate_paths)
{
    for (const auto &path : candidate_paths)
    {
        ifstream file(path, ios::binary);
        if (!file.is_open())
            continue;

        stringstream ss;
        ss << file.rdbuf();
        string raw = ss.str();
        if (raw.empty())
            continue;

        string combined;
        size_t i = 0;
        size_t n = raw.size();

        vector<string> current_row;
        string current_field;
        bool in_quotes = false;
        bool is_header = true;
        size_t rows_parsed = 0;

        while (i < n)
        {
            char c = raw[i];
            if (in_quotes)
            {
                if (c == '"')
                {
                    if (i + 1 < n && raw[i + 1] == '"')
                    {
                        current_field += '"';
                        i += 2;
                        continue;
                    }
                    else
                    {
                        in_quotes = false;
                        i++;
                        continue;
                    }
                }
                else
                {
                    current_field += c;
                    i++;
                    continue;
                }
            }
            else
            {
                if (c == '"')
                {
                    in_quotes = true;
                    i++;
                    continue;
                }
                else if (c == ',')
                {
                    current_row.push_back(current_field);
                    current_field.clear();
                    i++;
                    continue;
                }
                else if (c == '\r')
                {
                    if (i + 1 < n && raw[i + 1] == '\n')
                        i++;
                    current_row.push_back(current_field);
                    current_field.clear();

                    if (!is_header && current_row.size() >= 2)
                    {
                        string prompt = current_row[1];
                        string response = (current_row.size() >= 3) ? current_row[2] : "";
                        if (!response.empty())
                        {
                            combined += "// " + prompt + "\n" + response + "\n\n";
                        }
                        else
                        {
                            combined += prompt + "\n\n";
                        }
                        rows_parsed++;
                    }
                    is_header = false;
                    current_row.clear();
                    i++;
                    continue;
                }
                else if (c == '\n')
                {
                    current_row.push_back(current_field);
                    current_field.clear();

                    if (!is_header && current_row.size() >= 2)
                    {
                        string prompt = current_row[1];
                        string response = (current_row.size() >= 3) ? current_row[2] : "";
                        if (!response.empty())
                        {
                            combined += "// " + prompt + "\n" + response + "\n\n";
                        }
                        else
                        {
                            combined += prompt + "\n\n";
                        }
                        rows_parsed++;
                    }
                    is_header = false;
                    current_row.clear();
                    i++;
                    continue;
                }
                else
                {
                    current_field += c;
                    i++;
                    continue;
                }
            }
        }

        if (!current_field.empty() || !current_row.empty())
        {
            current_row.push_back(current_field);
            if (!is_header && current_row.size() >= 2)
            {
                string prompt = current_row[1];
                string response = (current_row.size() >= 3) ? current_row[2] : "";
                if (!response.empty())
                {
                    combined += "// " + prompt + "\n" + response + "\n\n";
                }
                rows_parsed++;
            }
        }

        cout << "  Loaded CSV code dataset from: " << path
             << " (" << rows_parsed << " programming exercises, " << combined.size() << " characters)\n";
        return combined;
    }
    return "";
}

// Loads natural language words database from data/corpus.txt and data/new_data/
static string load_words_database()
{
    string combined;

    // 1. Primary classical words corpus (corpus.txt)
    vector<string> corpus_candidates = {
        "data/corpus.txt",
        "../data/corpus.txt",
        "../../data/corpus.txt",
        "corpus.txt"};
    string corpus_txt = load_corpus_from_disk(corpus_candidates);
    if (!corpus_txt.empty())
    {
        cout << "  Loaded primary words corpus (data/corpus.txt): " << corpus_txt.size() << " characters\n";
        combined += corpus_txt;
        combined += "\n\n";
    }

    // 2. Load all textual and markdown documents from new_data/
    vector<string> new_data_dirs = {
        "data/new_data",
        "../data/new_data",
        "../../data/new_data",
        "new_data"};

    string found_dir;
    for (const auto &d : new_data_dirs)
    {
        if (filesystem::exists(d) && filesystem::is_directory(d))
        {
            found_dir = d;
            break;
        }
    }

    if (!found_dir.empty())
    {
        size_t new_data_files = 0;
        size_t new_data_bytes = 0;
        for (const auto &entry : filesystem::directory_iterator(found_dir))
        {
            if (!entry.is_regular_file())
                continue;
            string ext = entry.path().extension().string();
            // Ingest clean natural language text and markdown prose (skip raw tabular numerical CSVs and JSON schemas)
            if (ext == ".md" || ext == ".txt")
            {
                ifstream f(entry.path(), ios::binary);
                if (f.is_open())
                {
                    stringstream ss;
                    ss << f.rdbuf();
                    string content = ss.str();
                    new_data_bytes += content.size();
                    combined += content;
                    combined += "\n\n";
                    new_data_files++;
                }
            }
        }
        cout << "  Loaded new_data words database (" << found_dir << "): " << new_data_files
             << " files (" << new_data_bytes << " characters)\n";
    }

    return combined;
}

/**
 * @file main.cpp
 * @brief Demonstration application showcasing end-to-end tokenization, LLM training,
 *        and lightning-fast KV-Cached streaming autoregressive text generation.
 */
int main(int argc, char *argv[])
{
    cout << "=========================================================\n";
    cout << "    RINGWRAPPER CAUSAL TRANSFORMER LLM (GPT DECODER)     \n";
    cout << "=========================================================\n";
    cout << "Layered Architecture:\n";
    cout << "  [Ring 0] Tensor3D, LayerNorm, GELU, Causal Masking, Losses\n";
    cout << "  [Ring 1] Token/Pos Embeddings, Causal Attention, KV-Cache, AdamW\n";
    cout << "  [Ring 2] TransformerLM Decoder, Tokenizer & Autoregressive Gen\n";
    cout << "  [Ring 3] Text Next-Token Dataset & LLMTrainer\n";
    cout << "  [Ring 4] Interactive Streaming LLM Application\n";
    cout << "=========================================================\n\n";

    // CLI configuration options
    vector<string> cli_data_files;
    vector<string> cli_data_dirs;
    size_t cli_max_seq_len = 2048;
    size_t cli_init_seq_len = 64;
    size_t cli_steps = 25000;
    size_t cli_batch_size = 32;
    size_t cli_max_vocab_size = 10000; // Scalable up to 10k+ tokens
    float cli_lr = 0.35f;
    bool cli_debug = false;
    bool cli_safe_mode = false; // Phase 0: single-flag ablation baseline

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--data" || arg == "-d")
        {
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                cli_data_files.push_back(argv[++i]);
            }
        }
        else if (arg == "--dir")
        {
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                cli_data_dirs.push_back(argv[++i]);
            }
        }
        else if ((arg == "--vocab-size" || arg == "--max-vocab") && i + 1 < argc)
        {
            cli_max_vocab_size = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--seq-len" && i + 1 < argc)
        {
            cli_max_seq_len = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--init-seq-len" && i + 1 < argc)
        {
            cli_init_seq_len = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            cli_steps = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            cli_batch_size = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            cli_lr = stof(argv[++i]);
        }
        else if (arg == "--debug" || arg == "-v" || arg == "--verbose")
        {
            cli_debug = true;
        }
        else if (arg == "--safe-mode" || arg == "--safe")
        {
            // Phase 0 ablation baseline: disables meta-network, Taylor foresight
            // nudges, 4-formula routing, progressive depth/context/dataset growth,
            // Armijo/curvature scaling, and the info-entropy data filter.
            cli_safe_mode = true;
        }
        else if (arg == "--thought-loops" && i + 1 < argc)
        {
            ring0::get_config().default_thought_loops = static_cast<size_t>(stoul(argv[++i]));
        }
        else if (arg == "--reflection-cycles" && i + 1 < argc)
        {
            ring0::get_config().max_chain_reflection_cycles = static_cast<size_t>(stoul(argv[++i]));
        }
    }

    // Configure global runtime telemetry & debug mode
    ring0::get_config().set_debug(cli_debug);
    if (cli_debug)
    {
        cout << "  ⚙️ [Config] Debug Mode: ENABLED (Verbose Thought Chains & Diagnostics Active)\n\n";
    }

    // 1. Text & Multi-Format Ingestion (CSV, TXT, BIN, Code files)
    cout << "[Step 1] Initializing Universal Multi-Format Data Ingestion Engine (data/)...\n";
    string text_corpus;
    vector<int> binary_tokens;
    ring3::IngestionReport report;

    bool loaded_any = ring3::UniversalDataLoader::load_all_from_directory("data", text_corpus, binary_tokens, &report);
    if (!loaded_any || text_corpus.empty())
    {
        text_corpus = load_words_database();
    }

    // Ingest additional files specified via CLI
    for (const auto &fpath : cli_data_files)
    {
        if (filesystem::exists(fpath))
        {
            string ext = filesystem::path(fpath).extension().string();
            if (ext == ".csv" || ext == ".tsv")
            {
                size_t rows = 0;
                string csv_txt = ring3::UniversalDataLoader::parse_csv_to_text(fpath, &rows);
                text_corpus += csv_txt;
                report.csv_files_count++;
                report.total_csv_rows += rows;
                cout << "  CLI Ingested CSV file: " << fpath << " (" << rows << " rows)\n";
            }
            else if (ext == ".bin")
            {
                string bin_txt;
                size_t prev_toks = binary_tokens.size();
                ring3::UniversalDataLoader::parse_bin_file(fpath, bin_txt, binary_tokens);
                text_corpus += bin_txt;
                report.bin_files_count++;
                report.total_binary_tokens += (binary_tokens.size() - prev_toks);
                cout << "  CLI Ingested binary (.bin) file: " << fpath << "\n";
            }
            else
            {
                string txt = ring3::UniversalDataLoader::read_text_file(fpath);
                text_corpus += txt;
                report.txt_files_count++;
                cout << "  CLI Ingested text file: " << fpath << "\n";
            }
        }
    }

    // Ingest additional directories specified via CLI
    for (const auto &dpath : cli_data_dirs)
    {
        if (filesystem::exists(dpath) && filesystem::is_directory(dpath))
        {
            ring3::UniversalDataLoader::load_all_from_directory(dpath, text_corpus, binary_tokens, &report);
        }
    }

    if (text_corpus.empty())
    {
        cout << "  (No words database found; using inline fallback.)\n";
        text_corpus =
            "Once upon a time, in a peaceful land surrounded by green hills, "
            "there lived a kind king who cared for all creatures. "
            "The king had a wise wizard who could speak with birds and trees. "
            "Every morning, the wizard walked through the forest and listened to stories of the wind. "
            "One day, a young traveler arrived at the castle gates with a mysterious golden key. "
            "The traveler said, 'This key unlocks the ancient library hidden beneath the mountain.' "
            "The king smiled and welcomed the traveler into the kingdom. "
            "Together with the wizard, they set out on an adventure to discover ancient knowledge and magic. "
            "In the deep library, they found books filled with wonder, light, and friendship for all people.\n";
    }

    cout << "  📊 [Universal Data Ingestion Summary]\n";
    cout << "     • Total Files Discovered: " << report.total_files_count << " files across data/\n";
    cout << "     • Plain Text Documents (.txt, .md, .json): " << report.txt_files_count << " files\n";
    cout << "     • Tabular Datasets (.csv, .tsv): " << report.csv_files_count << " files (" << report.total_csv_rows << " structured records)\n";
    cout << "     • Source Code Files (.cpp, .hpp, .py): " << report.code_files_count << " files\n";
    cout << "     • Binary Files (.bin): " << report.bin_files_count << " files (" << report.total_binary_tokens << " raw tokens)\n";
    cout << "     • Combined Training Corpus: " << text_corpus.size() << " characters\n\n";

    ring2::Tokenizer tokenizer;
    cout << "  🧬 [Dynamic Factor-Based Token Sizing]\n";
    ring2::DynamicVocabFactors v_factors = tokenizer.fit_adaptive(text_corpus, cli_max_vocab_size);
    cout << "     • Corpus Volume Analyzed: " << (v_factors.corpus_bytes / 1024) << " KB (" << v_factors.corpus_bytes << " bytes)\n";
    cout << "     • Shannon Information Entropy: " << fixed << setprecision(3) << v_factors.shannon_entropy << " bits/char\n";
    cout << "     • Unique Alphabet Diversity: " << v_factors.unique_chars_count << " distinct base characters\n";
    cout << "     • Subword Compression Efficiency: " << fixed << setprecision(2) << v_factors.compression_ratio << " chars/token\n";
    cout << "     • Factor-Optimized Vocab Capacity: " << tokenizer.get_vocab_size() << " / " << cli_max_vocab_size << " subwords ("
         << tokenizer.merges.size() << " BPE merges extracted)\n\n";

    // 3. Dataset preparation
    size_t seq_len = cli_init_seq_len;
    ring3::TextDataset dataset(text_corpus, tokenizer, seq_len);
    dataset.sft_masking_enabled = false;

    // Append any pre-tokenized binary token IDs directly into the dataset token stream
    if (!binary_tokens.empty())
    {
        dataset.token_stream.insert(dataset.token_stream.end(), binary_tokens.begin(), binary_tokens.end());
        cout << "  📥 Appended " << binary_tokens.size() << " pre-tokenized binary token IDs directly to dataset stream!\n";
    }

    // Demonstrate Token Relevancy & Interpolated Neighborhood Context Parsing Algorithm
    cout << "  🎯 [Token Relevancy & Context Parsing Algorithm]\n";
    cout << "     Evaluating tokens with dynamic interpolated context parsing radius:\n";
    for (size_t sample_pos : {10, 45, 100})
    {
        if (sample_pos < dataset.token_stream.size())
        {
            auto win = dataset.extract_relevance_parsed_window(sample_pos);
            string token_str = tokenizer.decode({win.anchor_token});
            cout << "     • Anchor Token '" << token_str << "' (ID: " << win.anchor_token
                 << ", Pos: " << win.anchor_position << ") -> Relevancy Score: "
                 << fixed << setprecision(3) << win.anchor_relevance
                 << " | Parsed Window Radius: ±" << win.parsed_window_radius << " tokens ("
                 << win.tokens.size() << " tokens parsed with distance-decayed relevancies)\n";
        }
    }
    cout << "\n";

    // 4. Model configuration
    cout << "[Step 2] Constructing Causal Transformer LLM (GQA + SwiGLU + RoPE)...\n";
    ring0::CUDAMathEngine::initialize();
    const auto &dev_info = ring0::CUDAMathEngine::get_device_info();
    cout << "  🚀 [Math Hardware Engine] " << dev_info.device_name
         << " (" << dev_info.multi_processor_count << " parallel compute units active)\n";

    ring2::TransformerConfig lm_cfg;
    lm_cfg.vocab_size = tokenizer.get_vocab_size();
    lm_cfg.max_seq_len = cli_max_seq_len;
    lm_cfg.embed_dim = 128;
    lm_cfg.num_heads = 8;
    lm_cfg.num_kv_heads = 4; // Grouped-Query Attention (4 Query heads, 2 KV heads)
    lm_cfg.num_layers = 5;
    lm_cfg.ffn_dim = 256;

    ring2::TransformerLM model(lm_cfg);
    model.print_architecture();

    // Fast-Start: seed the LM-head bias with log-unigram frequencies so step-0
    // loss begins at the unigram entropy instead of the uniform ln(V) floor.
    // (Overwritten harmlessly if a better checkpoint is loaded below.)
    model.init_head_bias_from_unigram(dataset.token_stream);

    // Initialize Data-Agnostic Semantic VocabManager with 16 Vector Indexed Clusters and Layered Lexicon
    ring2::VocabManager vocab_mgr(lm_cfg.embed_dim, 16);
    vocab_mgr.initialize_base_meanings(tokenizer);
    vocab_mgr.parse_corpus_lexicon(text_corpus);

    // 5. Training Engine Setup & Baseline Loss Evaluation
    ring3::LLMTrainingConfig train_cfg;
    train_cfg.steps = cli_steps;
    train_cfg.batch_size = cli_batch_size;
    train_cfg.learning_rate = cli_lr;
    train_cfg.min_learning_rate = 0.0001f;
    train_cfg.warmup_ratio = 0.02f;
    train_cfg.weight_decay = 0.01f;
    train_cfg.eval_interval = 50;
    train_cfg.initial_seq_len = cli_init_seq_len;
    train_cfg.max_seq_len = cli_max_seq_len;
    train_cfg.step_based_context_growth = true;
    train_cfg.safe_mode = cli_safe_mode; // --safe-mode: override adaptive modules off
    // Start the curriculum on a larger slice (30% vs 5%): the tiny high-entropy
    // filtered slice made early per-step loss read artificially high while the
    // model was weakest. A wider slice better matches the corpus unigram bias.
    train_cfg.initial_dataset_ratio = 0.30f;

    ring3::LLMTrainer trainer(model, train_cfg);

    // Coupled Parameter & Vocabulary Expansion: Whenever model expands capacity, expand vocabulary too
    trainer.on_param_expansion = [&]()
    {
        cout << "\n  🧠 [Param Expansion Trigger] Expanding Vocabulary with Semantic Meaning Matrices!\n";
        vocab_mgr.scan_and_expand_vocabulary(text_corpus, tokenizer, model, 8);
    };

    // Evaluate current network baseline loss before deciding whether to use saved references
    float initial_eval_loss = trainer.evaluate_loss(dataset, 4);

    // Checkpoint Reference Loading: Only adopts saved data if its loss is around or less than current baseline
    const string checkpoint_path = "data/llm_checkpoint.bin";
    bool checkpoint_loaded = model.load_best_checkpoint_from_dir("checkpoints", initial_eval_loss);
    if (!checkpoint_loaded)
    {
        checkpoint_loaded = model.load_checkpoint_if_better(checkpoint_path, initial_eval_loss);
    }
    if (checkpoint_loaded)
    {
        cout << "  >> [Reference Status] Successfully adopted prior knowledge checkpoint into model!\n\n";
    }
    else
    {
        cout << "  >> [Reference Status] Training fresh from baseline initialization (current loss: "
             << fixed << setprecision(3) << initial_eval_loss << ").\n\n";
    }

    // 6. Baseline generation before training
    cout << "[Step 3] Baseline Word Generation (Prompt: 'The '):\n";
    string prompt_str = "The ";
    vector<int> prompt_tokens = tokenizer.encode(prompt_str);
    cout << "  Prompt: \"" << prompt_str << "\"\n  Streaming: \"" << prompt_str;
    model.generate(prompt_tokens, 40, 0.8f, 50, 0.70f, 0.05f, 1.2f, [&](int token)
                   {
        vector<int> t = {token};
        cout << tokenizer.decode(t) << flush; }, true);
    cout << "\"\n\n";

    // 7. Model Training Loop
    cout << "[Step 4] Training Transformer LLM on Words Database (corpus + new_data) with Dynamic LR Schedule & VocabManager...\n";

    // Prompts rotated through on each evaluation challenge.
    vector<string> challenge_prompts = {
        "The future of ",
        "To be, or not to ",
        "Large language models are ",
        "Once upon a time "};

    float best_checkpoint_loss = initial_eval_loss;

    auto start_train = chrono::high_resolution_clock::now();

    trainer.train(
        dataset,
        [&](const ring3::LLMStepMetrics &m)
        {
            if (cli_steps <= 10 || m.step % 10 == 0 || m.step == 1 || m.step == cli_steps)
            {
                cout << "  Step " << setw(5) << m.step << " / " << setw(5) << cli_steps
                     << " | Loss: " << fixed << setprecision(4) << m.loss
                     << " | LR: " << fixed << setprecision(6) << m.learning_rate
                     << " | Top-1: " << fixed << setprecision(1) << m.top1_accuracy << "%"
                     << " | Top-20: " << fixed << setprecision(1) << m.top20_accuracy << "%"
                     << " | PPL: " << fixed << setprecision(1) << m.perplexity
                     << " | Ctx: " << m.active_seq_len;
                if (trainer.watchdog_active)
                {
                    cout << " [RECOVERY]";
                }
                cout << "\n"
                     << flush;
            }

            // Continuous Multi-Layer Vocab Loop alongside training loop:
            // Loops through parsed lexicon, checks category losses, and dynamically expands tokens
            if (m.step % 15 == 0)
            {
                vocab_mgr.step_vocab_evaluation(text_corpus, tokenizer, model, m.loss);
            }

            // Save multi-file checkpoint bundle ONLY if loss is 5.2 or less
            if (m.step >= 50 && m.loss <= 5.2f && m.loss < (best_checkpoint_loss * 1.02f))
            {
                best_checkpoint_loss = min(best_checkpoint_loss, m.loss);

                // 1. Generate quick text snapshot
                string sample_str = "The future of ";
                vector<int> sample_p = tokenizer.encode("The future of ");
                model.generate(sample_p, 100, 0.55f, 40, 0.90f, 0.05f, 1.2f, [&](int tok)
                               {
                    vector<int> t = {tok};
                    sample_str += tokenizer.decode(t); }, true);

                // 2. Create versioned folder in checkpoints/
                stringstream ss;
                ss << "checkpoints/milestone_step_" << setfill('0') << setw(4) << m.step
                   << "_loss_" << fixed << setprecision(2) << m.loss;
                string bundle_dir = ss.str();

                bool ok = model.save_checkpoint_bundle(bundle_dir, m.step, m.loss, m.top1_accuracy, m.rank_score_top20, tokenizer, sample_str);
                if (ok && false)
                {
                    cout << "\n  ⭐ [Milestone Checkpoint] Quality Loss (" << fixed << setprecision(3) << m.loss
                         << " <= 5.2) achieved! Saved multi-file bundle to: " << bundle_dir << "/\n"
                         << "     ├── model_weights.bin     (Raw binary neural network weights)\n"
                         << "     ├── metadata.txt          (Step, Loss, Top-1 Acc, Rank Score, Parameters)\n"
                         << "     ├── vocab.txt             (BPE Token table and learned merge rules)\n"
                         << "     └── sample_generation.txt (Text output generated at this milestone)\n\n";
                }
            }

            // Periodic Auto-Purge: Every 20 steps, purge stale checkpoints exceeding min_loss * 1.15f
            if (m.step % 20 == 0 && best_checkpoint_loss < 6.0f)
            {
                ring2::TransformerLM::purge_stale_checkpoints("checkpoints", best_checkpoint_loss, 1.15f);
            }
        },
        [&](size_t step)
        {
            // Multi-Layer Vocabulary Telemetry & Evaluation
            auto vocab_metrics = vocab_mgr.evaluate_vocab_layers(model);
            cout << "\n  🧬 [Vocab Layers Telemetry @ step " << step << "]\n"
                 << "     ├── Layer 0 (Subwords):        " << vocab_metrics.layer0_subwords << "\n"
                 << "     ├── Layer 1 (Terms & Chunks):  " << vocab_metrics.layer1_words << " parsed items\n"
                 << "     ├── Layer 2 (Vector Clusters): " << vocab_metrics.layer2_categories << " data-agnostic latent vector indexes\n"
                 << "     ├── Vector Codebook Loss:      " << fixed << setprecision(4) << vocab_metrics.category_clustering_loss << "\n"
                 << "     └── Hash Alignment Score:      " << fixed << setprecision(4) << vocab_metrics.hash_alignment_score << "\n";

            size_t idx = (step / 500 >= 1) ? (step / 500 - 1) : 0;
            const string &prompt = challenge_prompts[idx % challenge_prompts.size()];
            vector<int> p_tokens = tokenizer.encode(prompt);

            cout << "\n--- Natural Language Word Challenge @ step " << step << " ---\n";
            cout << "Prompt: \"" << prompt << "\"\n";
            cout << "Output: \"" << prompt;
            model.generate(p_tokens, 40, 0.75f, 50, 0.90f, 0.05f, 1.2f, [&](int token)
                           {
                vector<int> t = {token};
                cout << tokenizer.decode(t) << flush; }, true);
            cout << "\"\n----------------------------------\n\n";
        });

    auto end_train = chrono::high_resolution_clock::now();
    chrono::duration<double> train_duration = end_train - start_train;
    cout << "\nTraining completed in " << fixed << setprecision(2) << train_duration.count() << " seconds.\n";

    // Save updated checkpoint ONLY if best achieved loss is 2.6 or less
    if (best_checkpoint_loss <= 5.2f)
    {
        bool checkpoint_saved = model.save_checkpoint(checkpoint_path, best_checkpoint_loss);
        if (checkpoint_saved)
        {
            cout << "  >> [Checkpoint] Saved trained weights (Quality Loss: " << fixed << setprecision(3) << best_checkpoint_loss
                 << " <= 5.2) to " << checkpoint_path << " for future sessions!\n";
        }
    }
    else
    {
        cout << "  >> [Checkpoint] Skipped saving checkpoint: best loss (" << fixed << setprecision(3) << best_checkpoint_loss
             << ") did not reach the <= 5.2 quality requirement.\n";
    }

    // 7. Interactive Real-Time Streaming Generation with GQA & Min-P/Top-P
    cout << "\n[Step 5] Real-Time Streaming C++ Code Generation (KV-Cached O(1) Decoding):\n";
    cout << "=========================================================\n";

    vector<string> test_prompts = {
        "void forward(",
        "class Matrix ",
        "for (size_t i = 0; i < ",
        "int main() {\n"};

    for (const auto &prompt : test_prompts)
    {
        vector<int> p_tokens = tokenizer.encode(prompt);
        cout << "Prompt: \"" << prompt << "\"\n";
        cout << "Output: \"" << prompt;

        auto t0 = chrono::high_resolution_clock::now();
        model.generate(p_tokens, 60, 0.75f, 50, 0.90f, 0.05f, 1.25f, [&](int token)
                       {
            vector<int> t = {token};
            cout << tokenizer.decode(t) << flush; }, true);
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> ms = t1 - t0;

        cout << "\"\n(Generated 60 tokens in " << fixed << setprecision(1) << ms.count() << " ms)\n";
        cout << "---------------------------------------------------------\n";
    }

    // 8. Benchmark: KV-Cached vs Non-Cached Generation Speed
    cout << "\n[Step 6] Performance Benchmark: KV-Cached vs Non-Cached Generation (100 tokens)\n";
    cout << "=========================================================\n";
    vector<int> bench_tokens = tokenizer.encode("Once upon a time ");

    // Without KV-Cache (Recomputes full prefix each step)
    auto b_start_uncached = chrono::high_resolution_clock::now();
    model.generate(bench_tokens, 100, 0.7f, 50, 0.90f, 0.05f, 1.2f, nullptr, false);
    auto b_end_uncached = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms_uncached = b_end_uncached - b_start_uncached;

    // With KV-Cache (O(1) step)
    auto b_start_cached = chrono::high_resolution_clock::now();
    model.generate(bench_tokens, 100, 0.7f, 50, 0.90f, 0.05f, 1.2f, nullptr, true);
    auto b_end_cached = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms_cached = b_end_cached - b_start_cached;

    cout << "  Without KV-Cache (O(T^2)): " << fixed << setprecision(2) << ms_uncached.count() << " ms\n";
    cout << "  With KV-Cache (O(1)):      " << fixed << setprecision(2) << ms_cached.count() << " ms\n";
    float speedup = ms_uncached.count() / max(0.001, ms_cached.count());
    cout << "  >> Speedup Factor:         " << fixed << setprecision(2) << speedup << "x faster!\n";
    cout << "=========================================================\n";

    // 9. Demonstration of Hierarchical Recursive AI Layer with Parent-Child Tree and Latent Thinking Loops
    cout << "\n[Step 7] Testing Hierarchical Recursive AI Layer (Thought Chains & Self-Reflection Looping)...\n";
    auto root_layer = make_unique<ring1::RecursiveLayer>("Root_Thought_Engine", 96, 96, 3);
    auto child_subthought = make_unique<ring1::RecursiveLayer>("Sub_Reasoner_A", 96, 96, 2);

    // Test parent pointer reachability
    root_layer->add_child(move(child_subthought));
    cout << "  Root Layer: " << root_layer->name << " (Thinking Depth: " << root_layer->thinking_depth
         << ", IsRoot: " << (root_layer->is_root() ? "true" : "false") << ")\n";
    cout << "  Child Layer: " << root_layer->children[0]->name
         << " (Tree Depth: " << root_layer->children[0]->get_tree_depth()
         << ", Reached Parent: " << (root_layer->children[0]->parent ? root_layer->children[0]->parent->name : "None") << ")\n\n";

    ring0::Matrix test_thought_input = ring0::Matrix::random_normal(1, 96, 0.0f, 0.1f);

    // Pass 1: Forward Reasoning through hierarchical Thought Chain
    cout << "--- Forward Thought Chain Reasoning ---\n";
    ring0::Matrix forward_thought = root_layer->forward(test_thought_input);
    cout << "  Thought Output Shape: (" << forward_thought.rows << "x" << forward_thought.cols << ")\n";

    // Pass 2: Multi-Pass Self-Reflection Looping through its own Thought Chains
    cout << "\n--- Multi-Pass Thought Chain Reflection Loops ---\n";
    size_t reflection_cycles = ring0::get_config().max_chain_reflection_cycles;
    ring0::Matrix refined_thought = root_layer->loop_thought_chain(test_thought_input, reflection_cycles, true);
    cout << "  Refined Thought Loop Output Shape: (" << refined_thought.rows << "x" << refined_thought.cols << ")\n";

    // Display Thought Chain Trace and Diagnostic Summary
    root_layer->print_thought_chain_summary();

    cout << "\nRingWrapper Causal Transformer LLM execution completed successfully!\n";
    return 0;
}
