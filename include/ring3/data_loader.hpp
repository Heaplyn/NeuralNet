#pragma once

/**
 * @file data_loader.hpp
 * @brief Universal Multi-Format Data Ingestion Engine for Ring 3.
 *        Recursively scans and parses .txt, .csv, .bin, and code files from data/.
 */

#include "ring2/tokenizer.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>

namespace ring3 {

class TextDataset;

/**
 * @struct IngestionReport
 * @brief Statistics on all ingested data files across data/ directory.
 */
struct IngestionReport {
    size_t txt_files_count = 0;
    size_t csv_files_count = 0;
    size_t bin_files_count = 0;
    size_t code_files_count = 0;
    size_t total_files_count = 0;

    size_t total_text_bytes = 0;
    size_t total_csv_rows = 0;
    size_t total_binary_tokens = 0;

    std::vector<std::string> loaded_file_paths;
};

/**
 * @class UniversalDataLoader
 * @brief Universal ingestion engine for structured, unstructured, and binary data.
 */
class UniversalDataLoader {
public:
    /**
     * @brief Recursively loads and parses all .txt, .csv, .bin, and code files from a directory.
     * @param directory_path Root data folder (e.g. "data")
     * @param out_corpus Combined text representation of all documents, CSV rows, and decoded streams
     * @param out_binary_tokens Pre-tokenized binary integer token IDs extracted from .bin files
     * @param report Ingestion statistics and summary
     * @return true if at least one file was successfully loaded
     */
    static bool load_all_from_directory(
        const std::string& directory_path,
        std::string& out_corpus,
        std::vector<int>& out_binary_tokens,
        IngestionReport* report = nullptr
    );

    /**
     * @brief Parses a CSV file into structured natural language records.
     *        Example: "[Record 1] Country: Canada | Literacy_Rate: 99.0% | Enrollment: 95.2%"
     */
    static std::string parse_csv_to_text(const std::string& csv_file_path, size_t* out_row_count = nullptr);

    /**
     * @brief Reads a plain text or code file into a string.
     */
    static std::string read_text_file(const std::string& file_path);

    /**
     * @brief Ingests binary .bin files (either as token IDs or UTF-8 / byte-stream representations).
     */
    static bool parse_bin_file(
        const std::string& bin_file_path,
        std::string& out_text,
        std::vector<int>& out_tokens
    );
};

/**
 * @class BackgroundDataStreamer
 * @brief Asynchronously streams and tokenizes data files in the background in real-time.
 *        Allows the LLM trainer to begin training immediately on an initial bootstrap chunk
 *        while additional files are discovered, loaded, and tokenized asynchronously without blocking training.
 */
class BackgroundDataStreamer {
private:
    std::vector<std::string> pending_files;
    size_t current_file_index = 0;
    std::atomic<bool> is_running{false};
    std::atomic<bool> all_done{false};
    std::thread worker_thread;
    mutable std::mutex stream_mutex;
    std::vector<int> buffered_tokens;
    std::string buffered_text;
    const ring2::Tokenizer* tokenizer_ref = nullptr;

    size_t total_streamed_tokens = 0;
    size_t total_streamed_files = 0;

public:
    BackgroundDataStreamer();
    ~BackgroundDataStreamer();

    // Discovers files in directory and enqueues them for background streaming
    bool initialize(const std::string& directory_path,
                    const ring2::Tokenizer& tokenizer,
                    size_t bootstrap_files = 1,
                    std::string* out_bootstrap_text = nullptr,
                    std::vector<int>* out_bootstrap_tokens = nullptr);

    // Starts background streaming worker thread
    void start();

    // Stops background streaming worker thread
    void stop();

    // Polls any newly tokenized background tokens and transfers them to TextDataset
    size_t poll_and_append(TextDataset& dataset);

    bool is_finished() const { return all_done.load(); }
    size_t get_total_streamed_tokens() const { return total_streamed_tokens; }
    size_t get_total_streamed_files() const { return total_streamed_files; }
    size_t get_total_enqueued_files() const { return pending_files.size(); }
};

} // namespace ring3
