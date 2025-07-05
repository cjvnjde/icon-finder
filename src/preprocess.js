import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";

const config = {
    iconSize: 64,
    iconsDir: "./icons",
    iconsPngDir: "./icons-png",
    iconDataFile: "./icon-data.json",
    featuresFile: "./features.json",
    version: "1.0.0",
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

        // 1. Convert SVG to PNG with a transparent background (restoring the original robust method)
        const pngBuffer = await sharp(Buffer.from(svgContent))
            .resize(config.iconSize, config.iconSize, {
                fit: "contain",
                background: {r: 255, g: 255, b: 255, alpha: 0}, // Transparent background
            })
            .png()
            .toBuffer();

        // Save the processed PNG file if enabled (useful for debugging)
        if (config.savePngFiles) {
            const pngFileName = `${iconName}.png`;
            const pngPath = path.join(config.iconsPngDir, pngFileName);
            fs.writeFileSync(pngPath, pngBuffer);
        }

        // 2. Process the PNG to get raw RGBA data
        const {data, info} = await sharp(pngBuffer)
            .raw()
            .toBuffer({resolveWithObject: true});

        const pixelArray = new Uint8Array(data);
        const expectedLength = config.iconSize ** 2;
        const channels = info.channels; // Should be 4 (RGBA)

        const processedArray = new Float32Array(expectedLength);

        // 3. Blend with a white background, then binarize and invert
        const threshold = 230; // Binarization threshold; pixels darker than this become the icon.

        for (let i = 0; i < expectedLength; i++) {
            const pixelStart = i * channels;
            const r = pixelArray[pixelStart];
            const g = pixelArray[pixelStart + 1];
            const b = pixelArray[pixelStart + 2];
            // Handle images that might not have an alpha channel
            const a = channels > 3 ? pixelArray[pixelStart + 3] : 255;

            // Convert to grayscale using the standard luminance formula
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Blend the grayscale pixel with a white background (255) using its alpha value
            const alpha = a / 255.0;
            const blendedGray = gray * alpha + 255 * (1 - alpha);

            // 4. Binarize and Invert the final pixel value
            // If the blended pixel is dark (part of the icon), set to 1.0.
            // If it's light (part of the background), set to 0.0.
            processedArray[i] = blendedGray < threshold ? 1.0 : 0.0;
        }

        // Convert Float32Array to a regular Array for JSON serialization
        const normalizedArray = Array.from(processedArray);

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
    if (config.savePngFiles) {
        console.log(`- PNG files saved to: ${config.iconsPngDir}`);
    }

    if (features.length !== iconData.length) {
        throw new Error("Mismatch between features and icon data lengths");
    }

    let minFeature = 1;
    let maxFeature = 0;
    let nonZeroFeatures = 0;
    let totalFeatures = 0;
    let significantFeatures = 0; // Features > 0.01

    for (const iconFeatures of features) {
        for (const value of iconFeatures) {
            if (value < minFeature) minFeature = value;
            if (value > maxFeature) maxFeature = value;
            if (value > 0) nonZeroFeatures++;
            if (value > 0.01) significantFeatures++;
            totalFeatures++;
        }
    }

    console.log(`Feature statistics:`);
    console.log(`- Min value: ${minFeature.toFixed(3)}`);
    console.log(`- Max value: ${maxFeature.toFixed(3)}`);
    console.log(`- Non-zero features: ${nonZeroFeatures}/${totalFeatures} (${(nonZeroFeatures / totalFeatures * 100).toFixed(1)}%)`);
    console.log(`- Significant features (>0.01): ${significantFeatures}/${totalFeatures} (${(significantFeatures / totalFeatures * 100).toFixed(1)}%)`);

    const metadata = {
        totalIcons: iconData.length,
        iconSize: config.iconSize,
        featureLength: config.iconSize ** 2,
        channels: 1,
        icons: iconData,
        processedAt: new Date().toISOString(),
        version: config.version,
        pngDirectoryPath: config.savePngFiles ? config.iconsPngDir : null,
        statistics: {
            minFeature,
            maxFeature,
            nonZeroFeatures,
            significantFeatures,
            totalFeatures
        }
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
    return [config.iconSize, config.iconSize, 1];
}

export function getFlattenedInputShape() {
    return [config.iconSize ** 2];
}
