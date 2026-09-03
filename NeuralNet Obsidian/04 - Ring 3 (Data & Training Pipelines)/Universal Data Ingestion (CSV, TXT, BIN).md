# 📂 Universal Multi-Format Data Ingestion (CSV, TXT, BIN, Code)

The `ring3::UniversalDataLoader` engine enables the neural network to automatically discover, parse, structure, and train on **any `.csv`, `.txt`, `.bin`, `.json`, `.md`, or `.cpp/.hpp` code files** placed inside the `data/` directory.

---

## 🎓 Beginner-Friendly Learning Guide: How Multi-Format Training Works

### The Data Ingestion Dilemma
In real-world deep learning, data comes in many different file formats:
- **Unstructured Text (`.txt`, `.md`)**: Books, articles, documentation, narrative stories.
- **Tabular Data (`.csv`, `.tsv`)**: Spreadsheets, databases, statistics (e.g. `Global_Education.csv`).
- **Source Code (`.cpp`, `.hpp`, `.py`)**: High-density syntax and algorithms.
- **Pre-Tokenized Binaries (`.bin`)**: Binary streams of 16-bit or 32-bit token IDs.

`UniversalDataLoader` scans the `data/` directory recursively and unifies all these heterogeneous data sources into a single rich training stream.

---

## 📊 Ingestion Pipeline Architecture

```mermaid
graph TD
    DataDir["data/ Directory (Recursive Scan)"] --> Router["File Extension Router"]
    
    Router -->|*.csv / *.tsv| CSV["CSVParser: Converts rows into structured '[Record N] Col: Val' sentences"]
    Router -->|*.txt / *.md / *.json| TXT["TextParser: Normalizes UTF-8 & adds document boundaries"]
    Router -->|*.cpp / *.hpp / *.py| CODE["CodeParser: Preserves indentation, comments, & syntax"]
    Router -->|*.bin / *.dat| BIN["BinaryParser: Ingests raw int32 token arrays or byte streams"]
    
    CSV --> Corpus["Unified Text Corpus (RAM)"]
    TXT --> Corpus
    CODE --> Corpus
    BIN --> BinaryStream["Pre-Tokenized Direct Stream"]
    
    Corpus --> BPE["BPE Subword Tokenizer (512 Vocab)"]
    BPE --> Dataset["TextDataset Token Stream"]
    BinaryStream --> Dataset
```

---

## 🔬 How Tabular CSV Ingestion Works

Standard language models cannot easily parse raw comma-separated values (`"Canada,99.0,95.2"`) because commas lack semantic context.

`UniversalDataLoader::parse_csv_to_text` reads column headers and transforms each row into an explicit natural language record:

```
=== [Tabular Dataset: data/Global_Education.csv | Columns: 5] ===
[Record 1] Countries: Canada | Youth_Literacy_Rate: 99.0% | Primary_Enrollment: 98.4% | Tertiary_Enrollment: 72.1%
[Record 2] Countries: Germany | Youth_Literacy_Rate: 99.5% | Primary_Enrollment: 99.1% | Tertiary_Enrollment: 70.3%
[Record 3] Countries: Japan | Youth_Literacy_Rate: 99.9% | Primary_Enrollment: 99.8% | Tertiary_Enrollment: 82.5%
```

> [!TIP]
> **Why this helps**: The transformer learns statistical correlations directly in English (e.g., that higher enrollment correlates with higher literacy rates).

---

## 💻 Deep Code Breakdown

Located in `src/ring3/data_loader.cpp`:

```cpp
bool UniversalDataLoader::load_all_from_directory(
    const std::string& directory_path,
    std::string& out_corpus,
    std::vector<int>& out_binary_tokens,
    IngestionReport* report
) {
    if (!std::filesystem::exists(directory_path)) return false;

    std::stringstream full_corpus;
    IngestionReport local_rep;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
        if (!entry.is_regular_file()) continue;

        auto path = entry.path();
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::string filepath_str = path.string();

        // 1. Plain text & documentation
        if (ext == ".txt" || ext == ".md" || ext == ".json") {
            std::string txt = read_text_file(filepath_str);
            full_corpus << txt;
            local_rep.txt_files_count++;
            local_rep.total_text_bytes += txt.size();
        }
        // 2. Tabular CSV records
        else if (ext == ".csv" || ext == ".tsv") {
            size_t row_count = 0;
            std::string csv_txt = parse_csv_to_text(filepath_str, &row_count);
            full_corpus << csv_txt;
            local_rep.csv_files_count++;
            local_rep.total_csv_rows += row_count;
        }
        // 3. Source code files
        else if (ext == ".cpp" || ext == ".hpp" || ext == ".py" || ext == ".c" || ext == ".h") {
            std::string code_txt = read_text_file(filepath_str);
            full_corpus << code_txt;
            local_rep.code_files_count++;
            local_rep.total_text_bytes += code_txt.size();
        }
        // 4. Binary token streams
        else if (ext == ".bin" || ext == ".dat") {
            std::string bin_txt;
            size_t prev_toks = out_binary_tokens.size();
            if (parse_bin_file(filepath_str, bin_txt, out_binary_tokens)) {
                full_corpus << bin_txt;
                local_rep.bin_files_count++;
                local_rep.total_binary_tokens += (out_binary_tokens.size() - prev_toks);
            }
        }
    }

    out_corpus += full_corpus.str();
    if (report) *report = local_rep;
    return true;
}
```

---

## 🔗 Related Notes
- [[04 - Ring 3 (Data & Training Pipelines)/Token Relevancy & Interpolated Parsing|Token Relevancy & Interpolated Parsing]]
- [[04 - Ring 3 (Data & Training Pipelines)/Progressive Curriculum & Horizon Growth|Progressive Curriculum & Horizon Growth]]
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]]
- [[Index|Return to Master Index]]
