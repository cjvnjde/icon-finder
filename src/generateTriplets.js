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
        const {hardNegativeMining = true, isValidation = false} = options;

        const anchorData = [];
        const positiveData = [];
        const negativeData = [];

        for (let i = 0; i < numTriplets; i++) {
            const anchorIdx = Math.floor(Math.random() * features.length);

            // --- IMPROVEMENT: A positive is ALWAYS an augmentation of the anchor. ---
            const positiveIdx = anchorIdx;

            let negativeIdx;

            // Find a hard negative: one that is visually similar to the anchor but not the same.
            if (hardNegativeMining) {
                const similarities = [];
                for (let j = 0; j < features.length; j++) {
                    if (anchorIdx === j) continue;
                    similarities.push({
                        index: j,
                        similarity: calculateSimilarity(features[anchorIdx], features[j]),
                    });
                }
                similarities.sort((a, b) => b.similarity - a.similarity);

                // Select a negative from the top N most similar icons.
                const topK = Math.min(20, similarities.length - 1);
                negativeIdx = similarities[Math.floor(Math.random() * topK)].index;
            } else {
                // Select a random negative.
                do {
                    negativeIdx = Math.floor(Math.random() * features.length);
                } while (negativeIdx === anchorIdx);
            }

            anchorData.push(features[anchorIdx]);
            positiveData.push(features[positiveIdx]);
            negativeData.push(features[negativeIdx]);
        }

        // --- IMPROVEMENT: Apply augmentation only during training ---
        const shouldAugment = !isValidation;

        const toTensor = (data) => tf.tensor3d(data, inputShape);

        const anchorTensors = anchorData.map(d => shouldAugment ? augmentImageTensor(toTensor(d)) : toTensor(d));
        const positiveTensors = positiveData.map(d => shouldAugment ? augmentImageTensor(toTensor(d)) : toTensor(d));
        const negativeTensors = negativeData.map(d => toTensor(d)); // No need to augment negatives

        const anchors = tf.stack(anchorTensors);
        const positives = tf.stack(positiveTensors);
        const negatives = tf.stack(negativeTensors);

        tf.dispose(anchorTensors);
        tf.dispose(positiveTensors);
        tf.dispose(negativeTensors);

        return {anchors, positives, negatives};
    });
}
