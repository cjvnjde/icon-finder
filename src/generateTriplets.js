import tf from "@tensorflow/tfjs-node";
import {augmentImageTensor} from "./augmentImageTensor.js";
import {getInputShape} from "./preprocess.js";

export function generateTriplets(features, numTriplets) {
    return tf.tidy(() => {
        const inputShape = getInputShape();
        console.log(`Generating ${numTriplets} triplets with augmentation...`);

        const anchorData = [];
        const positiveData = [];
        const negativeData = [];

        for (let i = 0; i < numTriplets; i++) {
            // Select anchor randomly
            const anchorIdx = Math.floor(Math.random() * features.length);

            // Positive: same image (will be augmented differently)
            const positiveIdx = anchorIdx;

            // Negative: different random image
            let negativeIdx = Math.floor(Math.random() * features.length);
            while (negativeIdx === anchorIdx) {
                negativeIdx = Math.floor(Math.random() * features.length);
            }

            anchorData.push(features[anchorIdx]);
            positiveData.push(features[positiveIdx]);
            negativeData.push(features[negativeIdx]);
        }

        // Create tensors with different augmentations for anchor/positive
        const anchorTensors = anchorData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor); // Different augmentation
        });

        const positiveTensors = positiveData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor); // Different augmentation
        });

        const negativeTensors = negativeData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        const anchors = tf.stack(anchorTensors);
        const positives = tf.stack(positiveTensors);
        const negatives = tf.stack(negativeTensors);

        // Clean up individual tensors
        anchorTensors.forEach(t => t.dispose());
        positiveTensors.forEach(t => t.dispose());
        negativeTensors.forEach(t => t.dispose());

        return {anchors, positives, negatives};
    });
}
