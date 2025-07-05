import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";

const config = {
    iconSize: 64,
    iconsDir: "./icons",
    iconsPngDir: "./icons-png",
    iconDataFile: "./icon-data.json",
    featuresFile: "./features.json",
    version: "2.0.0",
    savePngFiles: true,
};

function clearFiles() {
    console.log("Clearing files...");
    fs.rmSync(config.iconDataFile, {force: true});
    fs.rmSync(config.featuresFile, {force: true});
    fs.rmSync(config.iconsPngDir, {force: true, recursive: true});
}

function ensurePngDirectory() {
    if (config.savePngFiles && !fs.existsSync(config.iconsPngDir)) {
        fs.mkdirSync(config.iconsPngDir, {recursive: true});
        console.log(`Created PNG directory: ${config.iconsPngDir}`);
    }
}

async function preprocessIcon(filename, index) {
    const iconName = path.basename(filename, ".svg");
    const svgPath = path.join(config.iconsDir, filename);

    try {
        const svgContent = fs.readFileSync(svgPath, "utf8");

        // Convert SVG to PNG with proper anti-aliasing
        const pngBuffer = await sharp(Buffer.from(svgContent))
            .resize(config.iconSize, config.iconSize, {
                fit: "contain",
                background: {r: 255, g: 255, b: 255, alpha: 0},
            })
            .png()
            .toBuffer();

        // Save the processed PNG file if enabled
        if (config.savePngFiles) {
            const pngFileName = `${iconName}.png`;
            const pngPath = path.join(config.iconsPngDir, pngFileName);
            fs.writeFileSync(pngPath, pngBuffer);
        }

        // Process the PNG to get raw RGBA data
        const {data, info} = await sharp(pngBuffer)
            .raw()
            .toBuffer({resolveWithObject: true});

        const pixelArray = new Uint8Array(data);
        const expectedLength = config.iconSize ** 2;
        const channels = info.channels;

        // IMPROVED: Use grayscale values instead of binary
        const processedArray = new Float32Array(expectedLength);

        for (let i = 0; i < expectedLength; i++) {
            const pixelStart = i * channels;
            const r = pixelArray[pixelStart];
            const g = pixelArray[pixelStart + 1];
            const b = pixelArray[pixelStart + 2];
            const a = channels > 3 ? pixelArray[pixelStart + 3] : 255;

            // Convert to grayscale
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Blend with white background
            const alpha = a / 255.0;
            const blendedGray = gray * alpha + 255 * (1 - alpha);

            // IMPROVED: Normalize to [0, 1] range but keep grayscale values
            // Invert so that dark pixels (icon) are close to 1
            processedArray[i] = 1.0 - (blendedGray / 255.0);
        }

        // Apply edge detection to capture shape information
        const edgeFeatures = extractEdgeFeatures(processedArray, config.iconSize);

        // Combine original and edge features
        const combinedFeatures = new Float32Array(expectedLength * 2);
        combinedFeatures.set(processedArray, 0);
        combinedFeatures.set(edgeFeatures, expectedLength);

        // Convert to regular Array for JSON serialization
        const normalizedArray = Array.from(combinedFeatures);

        const iconData = {
            name: iconName,
            path: filename,
            pngPath: config.savePngFiles ? `${iconName}.png` : null,
            index,
        };

        return {iconData, features: normalizedArray};
    } catch (error) {
        console.error(`Error processing ${filename}:`, error);
        return null;
    }
}

// Simple edge detection using Sobel operator
function extractEdgeFeatures(pixels, size) {
    const edges = new Float32Array(size * size);

    // Sobel kernels
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for (let y = 1; y < size - 1; y++) {
        for (let x = 1; x < size - 1; x++) {
            let gx = 0, gy = 0;

            // Apply Sobel kernels
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * size + (x + kx);
                    const kernelIdx = (ky + 1) * 3 + (kx + 1);
                    gx += pixels[idx] * sobelX[kernelIdx];
                    gy += pixels[idx] * sobelY[kernelIdx];
                }
            }

            // Calculate edge magnitude
            const magnitude = Math.sqrt(gx * gx + gy * gy);
            edges[y * size + x] = Math.min(magnitude, 1.0);
        }
    }

    return edges;
}

export async function preprocessAllIcons() {
    console.log("Starting icon preprocessing...");

    clearFiles();
    ensurePngDirectory();

    const svgFiles = fs
        .readdirSync(config.iconsDir)
        .filter((file) => file.endsWith(".svg"));

    console.log(`Found ${svgFiles.length} SVG files`);

    const iconData = [];
    const features = [];
    let successCount = 0;
    let errorCount = 0;

    const batchSize = 100;
    for (let i = 0; i < svgFiles.length; i += batchSize) {
        const batch = svgFiles.slice(i, i + batchSize);
        const batchPromises = batch.map((filename, batchIndex) =>
            preprocessIcon(filename, successCount + batchIndex),
        );

        const batchResults = await Promise.allSettled(batchPromises);

        for (const result of batchResults) {
            if (result.status === "fulfilled" && result.value) {
                features.push(result.value.features);
                iconData.push(result.value.iconData);
                successCount++;
            } else {
                errorCount++;
                if (result.status === "rejected") {
                    console.error("Batch processing error:", result.reason);
                }
            }
        }

        console.log(
            `Processed ${Math.min(i + batchSize, svgFiles.length)}/${svgFiles.length} icons (${successCount} successful, ${errorCount} errors)`,
        );
    }

    console.log(`\nProcessing complete:`);
    console.log(`- Successfully processed: ${successCount} icons`);
    console.log(`- Errors: ${errorCount} icons`);

    const metadata = {
        totalIcons: iconData.length,
        iconSize: config.iconSize,
        featureLength: config.iconSize ** 2 * 2, // Original + edge features
        channels: 2, // Grayscale + edges
        icons: iconData,
        processedAt: new Date().toISOString(),
        version: config.version,
        pngDirectoryPath: config.savePngFiles ? config.iconsPngDir : null,
    };

    fs.writeFileSync(config.iconDataFile, JSON.stringify(metadata, null, 2));
    console.log(`Saved metadata to ${config.iconDataFile}`);

    fs.writeFileSync(config.featuresFile, JSON.stringify(features));
    console.log(`Saved features to ${config.featuresFile}`);

    return {metadata, features};
}

export function loadProcessedData() {
    const metadata = JSON.parse(fs.readFileSync(config.iconDataFile, "utf8"));
    const features = JSON.parse(fs.readFileSync(config.featuresFile, "utf8"));

    return {metadata, features};
}

export function getInputShape() {
    return [config.iconSize, config.iconSize, 2]; // 2 channels: grayscale + edges
}

export function getFlattenedInputShape() {
    return [config.iconSize ** 2 * 2];
}
