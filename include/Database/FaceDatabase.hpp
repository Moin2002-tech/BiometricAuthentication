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
     */
    struct FaceRecord {
        int id;                     ///< Unique identifier
        std::string name;           ///< Person's name
        std::vector<float> embedding; ///< 512-dimensional face embedding
        std::string photoPath;      ///< Path to saved photo file
    };

    /**
     * @brief FaceDatabase manages a persistent database of known faces.
     *
     * Stores face embeddings (512-dim vectors from ArcFace) along with
     * person names and photos. Supports adding new persons and finding
     * the closest match by cosine similarity.
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
         * @brief Add a new person to the database.
         * @param name    Person's name.
         * @param embedding 512-dim face embedding vector.
         * @param facePhoto The face image to save.
         * @return The assigned ID.
         */
        int addPerson(const std::string& name,
                      const std::vector<float>& embedding,
                      const cv::Mat& facePhoto);

        /**
         * @brief Find the closest match in the database using cosine similarity.
         * @param embedding  512-dim query embedding.
         * @param threshold  Minimum similarity threshold (0.0–1.0). Default 0.5.
         * @return Index of best match, or -1 if none above threshold.
         */
        int findMatch(const std::vector<float>& embedding,
                      float threshold = 0.5f) const;

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
        static constexpr uint32_t VERSION = 1;
        static constexpr int EMBEDDING_DIM = 512;

        std::string getPhotosDir() const;
    };

} // namespace Database

#endif //BIOMETRICAUTHENTICATION_FACEDATABASE_HPP