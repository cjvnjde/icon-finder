import tf from "@tensorflow/tfjs-node";
import {augmentImageTensor} from "./augmentImageTensor.js";
import {getInputShape} from "./preprocess.js";

/**
 * Calculate similarity between two feature vectors
 * @param {Array} features1 - First feature vector
 * @param {Array} features2 - Second feature vector
 * @returns {number} Cosine similarity
 */
function calculateSimilarity(features1, features2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < features1.length; i++) {
        dotProduct += features1[i] * features2[i];
        norm1 += features1[i] * features1[i];
        norm2 += features2[i] * features2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (norm1 * norm2);
}

/**
 * Generate triplets with smarter positive/negative selection
 * @param {Array} features - Array of feature vectors
 * @param {number} numTriplets - Number of triplets to generate
 * @param {Object} options - Generation options
 * @returns {Object} Anchors, positives, and negatives tensors
 */
export function generateTriplets(features, numTriplets, options = {}) {
    return tf.tidy(() => {
        const inputShape = getInputShape();
        const {
            useSimilaritySearch = true,
            numCandidates = 10,
            hardNegativeMining = true
        } = options;

        console.log(`Generating ${numTriplets} triplets with improved selection...`);

        const anchorData = [];
        const positiveData = [];
        const negativeData = [];

        // Pre-compute a similarity matrix for a subset of icons if using similarity search
        let similarityCache = new Map();
        if (useSimilaritySearch && features.length < 1000) {
            console.log("Pre-computing similarities for better positive selection...");
            for (let i = 0; i < Math.min(features.length, 100); i++) {
                const similarities = [];
                for (let j = 0; j < features.length; j++) {
                    if (i !== j) {
                        similarities.push({
                            index: j,
                            similarity: calculateSimilarity(features[i], features[j])
                        });
                    }
                }
                similarities.sort((a, b) => b.similarity - a.similarity);
                similarityCache.set(i, similarities);
            }
        }

        for (let i = 0; i < numTriplets; i++) {
            // Select anchor randomly
            const anchorIdx = Math.floor(Math.random() * features.length);

            let positiveIdx, negativeIdx;

            if (useSimilaritySearch && similarityCache.has(anchorIdx)) {
                // Use pre-computed similarities
                const similarities = similarityCache.get(anchorIdx);

                // Select positive from top similar icons (not identical)
                const topK = Math.min(5, similarities.length);
                positiveIdx = similarities[Math.floor(Math.random() * topK)].index;

                // Select negative with hard negative mining
                if (hardNegativeMining) {
                    // Select from middle range - not too similar, not too different
                    const midStart = Math.floor(similarities.length * 0.3);
                    const midEnd = Math.floor(similarities.length * 0.7);
                    const midIdx = midStart + Math.floor(Math.random() * (midEnd - midStart));
                    negativeIdx = similarities[midIdx].index;
                } else {
                    // Select from bottom similar icons
                    const bottomK = Math.min(10, similarities.length);
                    negativeIdx = similarities[similarities.length - 1 - Math.floor(Math.random() * bottomK)].index;
                }
            } else {
                // Fallback: Use augmented versions of same image as positive
                positiveIdx = anchorIdx;

                // Random negative selection with verification
                negativeIdx = Math.floor(Math.random() * features.length);
                let attempts = 0;
                while (negativeIdx === anchorIdx && attempts < 10) {
                    negativeIdx = Math.floor(Math.random() * features.length);
                    attempts++;
                }
            }

            anchorData.push(features[anchorIdx]);
            positiveData.push(features[positiveIdx]);
            negativeData.push(features[negativeIdx]);
        }

        // Create tensors with augmentation
        const anchorTensors = anchorData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        const positiveTensors = positiveData.map((data, idx) => {
            const tensor = tf.tensor3d(data, inputShape);
            // Apply stronger augmentation to positives when they're the same as anchor
            return augmentImageTensor(tensor);
        });

        const negativeTensors = negativeData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        const anchors = tf.stack(anchorTensors);
        const positives = tf.stack(positiveTensors);
        const negatives = tf.stack(negativeTensors);

        // Clean up
        anchorTensors.forEach(t => t.dispose());
        positiveTensors.forEach(t => t.dispose());
        negativeTensors.forEach(t => t.dispose());

        return {anchors, positives, negatives};
    });
}
