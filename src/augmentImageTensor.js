import tf from "@tensorflow/tfjs-node";

/**
 * Augment an image tensor with improved augmentation
 * @param {tf.Tensor3D} tensor - Image tensor
 * @returns {tf.Tensor3D} Augmented image tensor
 */
export function augmentImageTensor(tensor) {
    return tf.tidy(() => {
        let augmented = tensor.clone();

        // Random horizontal flip
        if (Math.random() > 0.5) {
            const flipped = tf.reverse(augmented, 1);
            augmented.dispose();
            augmented = flipped;
        }

        // Random brightness adjustment (less aggressive)
        if (Math.random() > 0.4) {
            const brightness = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
            const brightened = tf.clipByValue(tf.mul(augmented, brightness), 0, 1);
            augmented.dispose();
            augmented = brightened;
        }

        // Random contrast adjustment (less aggressive)
        if (Math.random() > 0.4) {
            const contrast = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
            const mean = tf.mean(augmented);
            const contrasted = tf.clipByValue(
                tf.add(tf.mul(tf.sub(augmented, mean), contrast), mean),
                0, 1
            );
            augmented.dispose();
            mean.dispose();
            augmented = contrasted;
        }

        // Add small amount of noise
        if (Math.random() > 0.6) {
            const noise = tf.randomNormal(augmented.shape, 0, 0.03);
            const noisy = tf.clipByValue(tf.add(augmented, noise), 0, 1);
            augmented.dispose();
            noise.dispose();
            augmented = noisy;
        }

        // Random small translation (improved)
        if (Math.random() > 0.7) {
            const [height, width, channels] = augmented.shape;
            if (height > 8 && width > 8) {
                const maxShift = Math.floor(Math.min(height, width) * 0.1);
                const shiftY = Math.floor((Math.random() - 0.5) * 2 * maxShift);
                const shiftX = Math.floor((Math.random() - 0.5) * 2 * maxShift);

                const startY = Math.max(0, -shiftY);
                const startX = Math.max(0, -shiftX);
                const endY = Math.min(height, height - shiftY);
                const endX = Math.min(width, width - shiftX);

                const cropped = tf.slice(
                    augmented,
                    [startY, startX, 0],
                    [endY - startY, endX - startX, channels]
                );
                const resized = tf.image.resizeBilinear(cropped, [height, width]);
                augmented.dispose();
                cropped.dispose();
                augmented = resized;
            }
        }

        return augmented;
    });
}
