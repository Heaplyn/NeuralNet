#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>

#include "ring3/data_loader.hpp"
#include "ring3/text_dataset.hpp"

namespace ring3 {

static inline std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(static_cast<unsigned char>(*start))) {
        start++;
    }
    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(start, end + 1);
}

// Parses a single CSV line with support for quoted strings containing commas
static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current += '"';
                i++; // Skip escaped quote
            } else {
                in_quotes = !in_quotes;
            }
        } else if ((c == ',' || c == '\t' || c == ';') && !in_quotes) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    fields.push_back(trim(current));
    return fields;
}

std::string UniversalDataLoader::parse_csv_to_text(const std::string& csv_file_path, size_t* out_row_count) {
    std::ifstream file(csv_file_path, std::ios::in);
    if (!file.is_open()) return {};

    std::stringstream output;
    std::string line;
    std::vector<std::string> headers;
    size_t row_index = 0;

    // Read header line
    if (std::getline(file, line)) {
        // Strip UTF-8 BOM if present
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }
        headers = parse_csv_line(line);
    }

    if (headers.empty()) return {};

    output << "\n\n=== [Tabular Dataset: " << csv_file_path << " | Columns: " << headers.size() << "] ===\n";

    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto values = parse_csv_line(line);
        if (values.empty()) continue;

        row_index++;
        output << "[Record " << row_index << "] ";

        for (size_t i = 0; i < headers.size() && i < values.size(); ++i) {
            if (!values[i].empty()) {
                output << headers[i] << ": " << values[i];
                if (i + 1 < headers.size() && i + 1 < values.size()) {
                    output << " | ";
                }
            }
        }
        output << "\n";
    }

    if (out_row_count) {
        *out_row_count = row_index;
    }

    return output.str();
}

std::string UniversalDataLoader::read_text_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) return {};

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string contents = buffer.str();
    if (contents.empty()) return {};

    std::stringstream formatted;
    formatted << "\n\n=== [Document: " << file_path << " (" << contents.size() << " bytes)] ===\n";
    formatted << contents << "\n";
    return formatted.str();
}

bool UniversalDataLoader::parse_bin_file(
    const std::string& bin_file_path,
    std::string& out_text,
    std::vector<int>& out_tokens
) {
    // Check if this is an LLM weight checkpoint file (skip model weight binaries)
    if (bin_file_path.find("checkpoint") != std::string::npos || bin_file_path.find("weights") != std::string::npos) {
        return false;
    }

    std::ifstream file(bin_file_path, std::ios::binary);
    if (!file.is_open()) return false;

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size <= 0) return false;

    // Check if the binary is a pre-tokenized int32/uint16 array
    // If file size is multiple of 4 and file contains valid token indices (< 100,000)
    if (file_size % sizeof(int32_t) == 0 && file_size < 100 * 1024 * 1024) {
        size_t count = file_size / sizeof(int32_t);
        std::vector<int32_t> buf(count);
        file.read(reinterpret_cast<char*>(buf.data()), file_size);

        bool looks_like_token_ids = true;
        for (size_t i = 0; i < std::min<size_t>(count, 100); ++i) {
            if (buf[i] < 0 || buf[i] > 100000) {
                looks_like_token_ids = false;
                break;
            }
        }

        if (looks_like_token_ids && count > 0) {
            for (auto id : buf) {
                out_tokens.push_back(static_cast<int>(id));
            }
            return true;
        }
    }

    // Otherwise, treat as raw binary data / byte stream representation
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw_bytes(std::min<std::streamsize>(file_size, 512 * 1024));
    file.read(reinterpret_cast<char*>(raw_bytes.data()), raw_bytes.size());

    std::stringstream ss;
    ss << "\n\n=== [Binary Data Stream: " << bin_file_path << " (" << file_size << " bytes)] ===\n";
    for (size_t i = 0; i < raw_bytes.size(); ++i) {
        // Output printable characters or hex representations
        if (std::isprint(raw_bytes[i]) || raw_bytes[i] == '\n' || raw_bytes[i] == ' ') {
            ss << static_cast<char>(raw_bytes[i]);
        } else {
            ss << " 0x" << std::hex << (int)raw_bytes[i] << std::dec << " ";
        }
    }
    ss << "\n";
    out_text += ss.str();
    return true;
}

bool UniversalDataLoader::load_all_from_directory(
    const std::string& directory_path,
    std::string& out_corpus,
    std::vector<int>& out_binary_tokens,
    IngestionReport* report
) {
    if (!std::filesystem::exists(directory_path)) {
        return false;
    }

    std::stringstream full_corpus;
    IngestionReport local_rep;

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
            if (!entry.is_regular_file()) continue;

            auto path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            std::string filepath_str = path.string();

            // 1. Plain Text files (.txt, .md, .json)
            if (ext == ".txt" || ext == ".md" || ext == ".json") {
                std::string txt = read_text_file(filepath_str);
                if (!txt.empty()) {
                    full_corpus << txt;
                    local_rep.txt_files_count++;
                    local_rep.total_text_bytes += txt.size();
                    local_rep.loaded_file_paths.push_back(filepath_str);
                }
            }
            // 2. CSV Tabular files (.csv, .tsv)
            else if (ext == ".csv" || ext == ".tsv") {
                size_t row_count = 0;
                std::string csv_txt = parse_csv_to_text(filepath_str, &row_count);
                if (!csv_txt.empty()) {
                    full_corpus << csv_txt;
                    local_rep.csv_files_count++;
                    local_rep.total_csv_rows += row_count;
                    local_rep.total_text_bytes += csv_txt.size();
                    local_rep.loaded_file_paths.push_back(filepath_str);
                }
            }
            // 3. Source code files (.cpp, .hpp, .c, .h, .py, .html, .js)
            else if (ext == ".cpp" || ext == ".hpp" || ext == ".c" || ext == ".h" || ext == ".py" || ext == ".js") {
                std::string code_txt = read_text_file(filepath_str);
                if (!code_txt.empty()) {
                    full_corpus << code_txt;
                    local_rep.code_files_count++;
                    local_rep.total_text_bytes += code_txt.size();
                    local_rep.loaded_file_paths.push_back(filepath_str);
                }
            }
            // 4. Binary token streams or binary files (.bin, .idx1-ubyte, .idx3-ubyte)
            else if (ext == ".bin" || ext == ".idx1-ubyte" || ext == ".idx3-ubyte" || ext == ".dat") {
                std::string bin_txt;
                size_t prev_tok_count = out_binary_tokens.size();
                if (parse_bin_file(filepath_str, bin_txt, out_binary_tokens)) {
                    if (!bin_txt.empty()) {
                        full_corpus << bin_txt;
                        local_rep.total_text_bytes += bin_txt.size();
                    }
                    local_rep.total_binary_tokens += (out_binary_tokens.size() - prev_tok_count);
                    local_rep.bin_files_count++;
                    local_rep.loaded_file_paths.push_back(filepath_str);
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Warning during directory scan: " << ex.what() << "\n";
    }

    local_rep.total_files_count = local_rep.txt_files_count + local_rep.csv_files_count + 
                                  local_rep.code_files_count + local_rep.bin_files_count;

    out_corpus += full_corpus.str();

    if (report) {
        *report = local_rep;
    }

    return (local_rep.total_files_count > 0);
}

// --- Background Data Streamer Implementation ---

BackgroundDataStreamer::BackgroundDataStreamer()
    : is_running(false), all_done(false), current_file_index(0),
      total_streamed_tokens(0), total_streamed_files(0), tokenizer_ref(nullptr) {}

BackgroundDataStreamer::~BackgroundDataStreamer() {
    stop();
}

bool BackgroundDataStreamer::initialize(
    const std::string& directory_path,
    const ring2::Tokenizer& tokenizer,
    size_t bootstrap_files,
    std::string* out_bootstrap_text,
    std::vector<int>* out_bootstrap_tokens
) {
    tokenizer_ref = &tokenizer;
    pending_files.clear();
    current_file_index = 0;
    total_streamed_tokens = 0;
    total_streamed_files = 0;
    all_done.store(false);

    if (!std::filesystem::exists(directory_path) || !std::filesystem::is_directory(directory_path)) {
        all_done.store(true);
        return false;
    }

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
            if (!entry.is_regular_file()) continue;
            auto path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".txt" || ext == ".md" || ext == ".json" ||
                ext == ".csv" || ext == ".tsv" ||
                ext == ".cpp" || ext == ".hpp" || ext == ".c" || ext == ".h" || ext == ".py" || ext == ".js" ||
                ext == ".bin") {
                pending_files.push_back(path.string());
            }
        }
    } catch (...) {}

    if (pending_files.empty()) {
        all_done.store(true);
        return false;
    }

    // Process initial bootstrap files synchronously so training and tokenizer start immediately
    size_t bootstrap_count = std::min(bootstrap_files, pending_files.size());
    std::stringstream boot_ss;
    for (size_t i = 0; i < bootstrap_count; ++i) {
        const std::string& fpath = pending_files[i];
        std::string ext = std::filesystem::path(fpath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv" || ext == ".tsv") {
            size_t rows = 0;
            boot_ss << UniversalDataLoader::parse_csv_to_text(fpath, &rows) << "\n";
        } else if (ext == ".bin" && out_bootstrap_tokens) {
            std::string bin_txt;
            UniversalDataLoader::parse_bin_file(fpath, bin_txt, *out_bootstrap_tokens);
            if (!bin_txt.empty()) boot_ss << bin_txt << "\n";
        } else {
            boot_ss << UniversalDataLoader::read_text_file(fpath) << "\n";
        }
        total_streamed_files++;
    }

    if (out_bootstrap_text) {
        *out_bootstrap_text = boot_ss.str();
    }

    current_file_index = bootstrap_count;
    if (current_file_index >= pending_files.size()) {
        all_done.store(true);
    }

    return true;
}

void BackgroundDataStreamer::start() {
    if (is_running.load() || all_done.load()) return;
    is_running.store(true);

    worker_thread = std::thread([this]() {
        while (is_running.load() && current_file_index < pending_files.size()) {
            std::string fpath = pending_files[current_file_index++];
            std::string ext = std::filesystem::path(fpath).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            std::string file_text;
            std::vector<int> file_tokens;

            if (ext == ".csv" || ext == ".tsv") {
                size_t rows = 0;
                file_text = UniversalDataLoader::parse_csv_to_text(fpath, &rows);
            } else if (ext == ".bin") {
                UniversalDataLoader::parse_bin_file(fpath, file_text, file_tokens);
            } else {
                file_text = UniversalDataLoader::read_text_file(fpath);
            }

            if (!file_text.empty() && tokenizer_ref) {
                std::vector<int> encoded = tokenizer_ref->encode(file_text);
                file_tokens.insert(file_tokens.end(), encoded.begin(), encoded.end());
            }

            if (!file_tokens.empty()) {
                std::lock_guard<std::mutex> lock(stream_mutex);
                buffered_tokens.insert(buffered_tokens.end(), file_tokens.begin(), file_tokens.end());
                total_streamed_tokens += file_tokens.size();
                total_streamed_files++;
            }

            // Yield small duration to yield I/O and CPU to training worker threads
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        all_done.store(true);
        is_running.store(false);
    });
}

void BackgroundDataStreamer::stop() {
    is_running.store(false);
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

size_t BackgroundDataStreamer::poll_and_append(TextDataset& dataset) {
    std::vector<int> incoming;
    {
        std::lock_guard<std::mutex> lock(stream_mutex);
        if (buffered_tokens.empty()) return 0;
        incoming.swap(buffered_tokens);
    }
    size_t count = dataset.append_tokens(incoming);
    return count;
}

} // namespace ring3
