//
// Created by moinshaikh on 7/2/26.
//

#pragma once

#ifndef BIOMETRICAUTHENTICATION_FACEDATABASE_HPP
#define BIOMETRICAUTHENTICATION_FACEDATABASE_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace Database {

    /**
     * @brief Represents a single person's face record stored in the database.
     *
     * Each person can have MULTIPLE embeddings (from different angles/lighting conditions)
     * and corresponding photos, making recognition more robust for side/angled views.
     */
    struct FaceRecord {
        int id;                     ///< Unique identifier
        std::string name;           ///< Person's name
        std::vector<std::vector<float>> embeddings; ///< 512-dim face embeddings (one per angle)
        std::vector<std::string> photoPaths;        ///< Paths to saved photo files (one per angle)
    };

    /**
     * @brief FaceDatabase manages a persistent database of known faces.
     *
     * Stores multiple face embeddings (512-dim vectors from ArcFace) per person
     * along with person names and photos. Supports adding new persons, adding
     * additional faces to existing persons, and finding the closest match
     * across all registered angles by cosine similarity.
     *
     * Data is stored in:
     *   - databases/faces.bin  : binary file with embeddings & metadata
     *   - databases/photos/     : directory with JPEG face crops
     */
    class FaceDatabase {
    public:
        /// Constructor: loads existing database from disk
        FaceDatabase();

        /// Destructor: saves database to disk
        ~FaceDatabase();

        /**
         * @brief Add a new person to the database (auto-assigned ID).
         * @param name    Person's name.
         * @param embedding 512-dim face embedding vector.
         * @param facePhoto The face image to save.
         * @return The assigned ID.
         */
        int addPerson(const std::string& name,
                      const std::vector<float>& embedding,
                      const cv::Mat& facePhoto);

        /**
         * @brief Add a new person with a manually specified ID.
         * @param id      Manual ID for better authorization control.
         * @param name    Person's name.
         * @param embedding 512-dim face embedding vector.
         * @param facePhoto The face image to save.
         * @return The manually-specified ID, or -1 if ID already exists.
         */
        int addPerson(int id,
                      const std::string& name,
                      const std::vector<float>& embedding,
                      const cv::Mat& facePhoto);

        /**
         * @brief Add an additional face (embedding + photo) to an existing person.
         * @param id        The person's ID.
         * @param embedding 512-dim face embedding vector (from a different angle).
         * @param facePhoto The face image to save.
         * @return true if successful, false if person ID not found.
         */
        bool addFaceToPerson(int id,
                             const std::vector<float>& embedding,
                             const cv::Mat& facePhoto);

        /**
         * @brief Check if a person with the given ID already exists.
         * @param id  The ID to check.
         * @return true if the ID is taken, false otherwise.
         */
        bool hasId(int id) const;

        /**
         * @brief Find the closest person match in the database using cosine similarity.
         *
         * Compares the query embedding against ALL stored embeddings for each person
         * and returns the person with the highest similarity score (best-of-N approach).
         *
         * @param embedding  512-dim query embedding.
         * @param threshold  Minimum similarity threshold (0.0–1.0). Default 0.35.
         * @return Index of best person match, or -1 if none above threshold.
         */
        int findMatch(const std::vector<float>& embedding,
                      float threshold = 0.35f) const;

        /**
         * @brief Get the best similarity score and person index for a given embedding.
         * @param embedding  512-dim query embedding.
         * @param bestScore  [out] The highest similarity score found.
         * @param threshold  Minimum similarity threshold.
         * @return Index of best person match, or -1 if none above threshold.
         */
        int findMatchWithScore(const std::vector<float>& embedding,
                               float& bestScore,
                               float threshold = 0.35f) const;

        /**
         * @brief Get cosine similarity between two embeddings.
         */
        static float cosineSimilarity(const std::vector<float>& a,
                                      const std::vector<float>& b);

        /**
         * @brief Get a FaceRecord by index.
         */
        const FaceRecord& getRecord(int index) const;

        /**
         * @brief Get the number of stored persons.
         */
        int size() const { return static_cast<int>(records_.size()); }

        /**
         * @brief Get the number of total face photos across all persons.
         */
        int totalFaces() const;

        /**
         * @brief Get the name of the person at index.
         */
        std::string getName(int index) const;

        /**
         * @brief Reset the database: clear all records, delete stored photos,
         *        and overwrite the binary file with an empty state.
         */
        void reset();

        /**
         * @brief Save database to disk.
         */
        void save();

        /**
         * @brief Load database from disk.
         */
        void load();

    private:
        std::vector<FaceRecord> records_;
        int nextId_ = 1;

        static constexpr const char* DB_DIR = "databases";
        static constexpr const char* DB_FILE = "databases/faces.bin";
        static constexpr const char* PHOTOS_DIR = "databases/photos";
        static constexpr uint32_t MAGIC = 0x46414345; // "FACE"
        static constexpr uint32_t VERSION = 2;        // Bumped to v2 for multi-embedding support
        static constexpr int EMBEDDING_DIM = 512;

        std::string getPhotosDir() const;
    };

} // namespace Database

#endif //BIOMETRICAUTHENTICATION_FACEDATABASE_HPP