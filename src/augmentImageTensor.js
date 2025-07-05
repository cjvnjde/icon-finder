import tf from "@tensorflow/tfjs-node";

/**
 * Augment an image tensor with improved augmentation for icon/drawing recognition
 * @param {tf.Tensor4D} tensor - Image tensor (batch, height, width, channels)
 * @returns {tf.Tensor4D} Augmented image tensor
 */
export function augmentImageTensor(tensor) {
    return tf.tidy(() => {
        // Ensure we have a 4D tensor (add batch dimension if needed)
        let augmented = tensor.rank === 3 ? tf.expandDims(tensor, 0) : tensor.clone();
        const [batchSize, height, width, channels] = augmented.shape;

        // Random horizontal flip (good for icons/drawings)
        if (Math.random() > 0.5) {
            augmented = tf.reverse(augmented, 2); // axis 2 for width in 4D tensor
        }

        // Random vertical flip (sometimes useful for icons)
        if (Math.random() > 0.7) {
            augmented = tf.reverse(augmented, 1); // axis 1 for height in 4D tensor
        }

        // Random brightness adjustment (conservative for drawings)
        if (Math.random() > 0.4) {
            const brightness = 0.85 + Math.random() * 0.3; // 0.85 to 1.15
            augmented = tf.clipByValue(tf.mul(augmented, brightness), 0, 1);
        }

        // Random contrast adjustment (helpful for varying line thickness)
        if (Math.random() > 0.4) {
            const contrast = 0.85 + Math.random() * 0.3; // 0.85 to 1.15
            const mean = tf.mean(augmented, [1, 2, 3], true); // Keep dims for broadcasting
            augmented = tf.clipByValue(
                tf.add(tf.mul(tf.sub(augmented, mean), contrast), mean),
                0, 1
            );
        }

        // Gaussian blur (simulate drawing variations)
        if (Math.random() > 0.8) {
            // Simple blur using average pooling as alternative to gaussian
            const blurred = tf.avgPool(augmented, 3, 1, 'same');
            const blendFactor = 0.3; // Mild blending
            augmented = tf.add(
                tf.mul(augmented, 1 - blendFactor),
                tf.mul(blurred, blendFactor)
            );
        }

        // Add subtle noise (simulate drawing imperfections)
        if (Math.random() > 0.6) {
            const noiseStrength = 0.02; // Very subtle for drawings
            const noise = tf.randomNormal(augmented.shape, 0, noiseStrength);
            augmented = tf.clipByValue(tf.add(augmented, noise), 0, 1);
        }

        // Random rotation (using coordinate-based approach)
        if (Math.random() > 0.7) {
            const maxAngle = Math.PI / 12; // ±15 degrees
            const angle = (Math.random() - 0.5) * 2 * maxAngle;

            if (Math.abs(angle) > 0.01) { // Only rotate if angle is significant
                const rotated = rotateImage(augmented, angle);
                augmented = rotated;
            }
        }

        // Random small translation using crop and resize (more reliable than transform)
        if (Math.random() > 0.6 && height > 16 && width > 16) {
            const maxShift = Math.floor(Math.min(height, width) * 0.08); // Smaller shifts
            const shiftY = Math.floor((Math.random() - 0.5) * 2 * maxShift);
            const shiftX = Math.floor((Math.random() - 0.5) * 2 * maxShift);

            // Calculate crop bounds
            const cropY = Math.max(0, shiftY);
            const cropX = Math.max(0, shiftX);
            const cropHeight = height - Math.abs(shiftY);
            const cropWidth = width - Math.abs(shiftX);

            if (cropHeight > height * 0.8 && cropWidth > width * 0.8) {
                const cropped = tf.slice(
                    augmented,
                    [0, cropY, cropX, 0],
                    [batchSize, cropHeight, cropWidth, channels]
                );
                augmented = tf.image.resizeBilinear(cropped, [height, width]);
            }
        }

        // Random zoom (crop center and resize)
        if (Math.random() > 0.8) {
            const zoomFactor = 0.9 + Math.random() * 0.2; // 0.9 to 1.1
            if (zoomFactor !== 1.0) {
                const newHeight = Math.floor(height * zoomFactor);
                const newWidth = Math.floor(width * zoomFactor);

                if (zoomFactor < 1.0) {
                    // Zoom out: resize then pad
                    const resized = tf.image.resizeBilinear(augmented, [newHeight, newWidth]);
                    const padY = Math.floor((height - newHeight) / 2);
                    const padX = Math.floor((width - newWidth) / 2);
                    const paddings = [[0, 0], [padY, height - newHeight - padY], [padX, width - newWidth - padX], [0, 0]];
                    augmented = tf.pad(resized, paddings, 0); // Pad with black
                } else {
                    // Zoom in: crop center then resize
                    const cropY = Math.floor((newHeight - height) / 2);
                    const cropX = Math.floor((newWidth - width) / 2);
                    const resized = tf.image.resizeBilinear(augmented, [newHeight, newWidth]);
                    augmented = tf.slice(resized, [0, cropY, cropX, 0], [batchSize, height, width, channels]);
                }
            }
        }

        // Intensity inversion (sometimes helpful for drawings)
        if (Math.random() > 0.9) {
            augmented = tf.sub(1, augmented);
        }

        // Ensure output maintains the original batch dimension expectation
        return tensor.rank === 3 ? tf.squeeze(augmented, 0) : augmented;
    });
}

/**
 * Rotate an image tensor using coordinate-based sampling (no Transform kernel needed)
 * @param {tf.Tensor4D} imageTensor - 4D image tensor
 * @param {number} angleRadians - Rotation angle in radians
 * @returns {tf.Tensor4D} Rotated image tensor
 */
function rotateImage(imageTensor, angleRadians) {
    return tf.tidy(() => {
        const [batchSize, height, width, channels] = imageTensor.shape;

        // For very small angles, use shear approximation
        if (Math.abs(angleRadians) < Math.PI / 72) { // Less than 2.5 degrees
            return applyShearRotation(imageTensor, angleRadians);
        }

        // For larger angles, use multi-step approximation
        return applyMultiStepRotation(imageTensor, angleRadians);
    });
}

/**
 * Apply rotation using shear approximation for very small angles
 * @param {tf.Tensor4D} imageTensor - 4D image tensor
 * @param {number} angleRadians - Small rotation angle
 * @returns {tf.Tensor4D} Shear-rotated image tensor
 */
function applyShearRotation(imageTensor, angleRadians) {
    return tf.tidy(() => {
        const [batchSize, height, width, channels] = imageTensor.shape;

        // Horizontal shear based on y-coordinate
        const shearFactor = Math.tan(angleRadians / 2);
        const maxShift = Math.floor(Math.abs(shearFactor) * height / 2);

        if (maxShift === 0) {
            return imageTensor.clone();
        }

        // Create shifted versions and blend
        const shifts = [];
        const weights = [];

        for (let y = 0; y < height; y++) {
            const shift = Math.round(shearFactor * (y - height / 2));
            if (!shifts.includes(shift)) {
                shifts.push(shift);
            }
        }

        // Limit to reasonable number of shifts
        const limitedShifts = shifts.slice(-3, 3);
        const shiftedImages = [];

        for (const shift of limitedShifts) {
            if (Math.abs(shift) < width * 0.2) {
                const startX = Math.max(0, -shift);
                const endX = Math.min(width, width - shift);
                const cropWidth = endX - startX;

                if (cropWidth > width * 0.7) {
                    const cropped = tf.slice(
                        imageTensor,
                        [0, 0, startX, 0],
                        [batchSize, height, cropWidth, channels]
                    );
                    const resized = tf.image.resizeBilinear(cropped, [height, width]);
                    shiftedImages.push(resized);
                }
            }
        }

        if (shiftedImages.length > 0) {
            // Average the shifted images for smoother result
            const averaged = tf.mean(tf.stack(shiftedImages), 0);
            return averaged;
        }

        return imageTensor.clone();
    });
}

/**
 * Apply rotation using multiple small transformations
 * @param {tf.Tensor4D} imageTensor - 4D image tensor
 * @param {number} angleRadians - Rotation angle
 * @returns {tf.Tensor4D} Rotated image tensor
 */
function applyMultiStepRotation(imageTensor, angleRadians) {
    return tf.tidy(() => {
        const [batchSize, height, width, channels] = imageTensor.shape;

        // Break rotation into multiple steps for better approximation
        const steps = Math.min(4, Math.ceil(Math.abs(angleRadians) / (Math.PI / 36))); // Max 4 steps
        const stepAngle = angleRadians / steps;

        let result = imageTensor.clone();

        for (let step = 0; step < steps; step++) {
            // Alternate between horizontal and vertical shifts to approximate rotation
            if (step % 2 === 0) {
                // Horizontal shift based on angle
                const shiftAmount = Math.sin(stepAngle) * (height / 4);
                const shift = Math.round(shiftAmount);

                if (Math.abs(shift) > 0 && Math.abs(shift) < width * 0.15) {
                    const startX = Math.max(0, -shift);
                    const endX = Math.min(width, width - shift);
                    const cropWidth = endX - startX;

                    if (cropWidth > width * 0.8) {
                        const cropped = tf.slice(
                            result,
                            [0, 0, startX, 0],
                            [batchSize, height, cropWidth, channels]
                        );
                        const newResult = tf.image.resizeBilinear(cropped, [height, width]);
                        result.dispose();
                        result = newResult;
                    }
                }
            } else {
                // Vertical shift
                const shiftAmount = Math.sin(stepAngle) * (width / 4);
                const shift = Math.round(shiftAmount);

                if (Math.abs(shift) > 0 && Math.abs(shift) < height * 0.15) {
                    const startY = Math.max(0, -shift);
                    const endY = Math.min(height, height - shift);
                    const cropHeight = endY - startY;

                    if (cropHeight > height * 0.8) {
                        const cropped = tf.slice(
                            result,
                            [0, startY, 0, 0],
                            [batchSize, cropHeight, width, channels]
                        );
                        const newResult = tf.image.resizeBilinear(cropped, [height, width]);
                        result.dispose();
                        result = newResult;
                    }
                }
            }
        }

        return result;
    });
}
