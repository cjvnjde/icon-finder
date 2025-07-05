import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";

const config = {
  iconSize: 64,
  iconsDir: "./icons",
  iconDataFile: "./icon-data.json",
  featuresFile: "./features.json",
  version: "1.0.0",
};

function clearFiles() {
  console.log("Clearing files...");
  fs.rmSync(config.iconDataFile, { force: true });
  fs.rmSync(config.featuresFile, { force: true });
}

async function preprocessIcon(filename, index) {
  const iconName = path.basename(filename, ".svg");
  const svgPath = path.join(config.iconsDir, filename);

  try {
    const svgContent = fs.readFileSync(svgPath, "utf8");

    const pngBuffer = await sharp(Buffer.from(svgContent))
      .resize(config.iconSize, config.iconSize, {
        fit: "contain",
        background: { r: 255, g: 255, b: 255, alpha: 0 },
      })
      .greyscale()
      .raw()
      .toBuffer();

    const pixelArray = new Uint8Array(pngBuffer);
    const expectedLength = config.iconSize ** 2;

    if (pixelArray.length !== expectedLength) {
      console.warn(
        `Warning: ${filename} has ${pixelArray.length} pixels, expected ${expectedLength}`,
      );
      return null;
    }

    const normalizedArray = Array.from(pixelArray).map((val) => val / 255.0);

    const iconData = {
      name: iconName,
      path: filename,
      index,
    };

    return { iconData, features: normalizedArray };
  } catch (error) {
    console.error(`Error processing ${filename}:`, error);
    return null;
  }
}

async function preprocessAllIcons() {
  console.log("Starting icon preprocessing...");

  clearFiles();

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
      }
    }

    console.log(
      `Processed ${Math.min(i + batchSize, svgFiles.length)}/${svgFiles.length} icons (${successCount} successful, ${errorCount} errors)`,
    );
  }

  console.log(`\nProcessing complete:`);
  console.log(`- Successfully processed: ${successCount} icons`);
  console.log(`- Errors: ${errorCount} icons`);

  if (features.length !== iconData.length) {
    throw new Error("Mismatch between features and icon data lengths");
  }

  const metadata = {
    totalIcons: iconData.length,
    iconSize: config.iconSize,
    featureLength: config.iconSize ** 2,
    channels: 1,
    icons: iconData,
    processedAt: new Date().toISOString(),
    version: config.version,
  };

  fs.writeFileSync(config.iconDataFile, JSON.stringify(metadata, null, 2));
  console.log(`Saved metadata to ${config.iconDataFile}`);

  fs.writeFileSync(config.featuresFile, JSON.stringify(features));
  console.log(`Saved features to ${config.featuresFile}`);

  return { metadata, features };
}

export function loadProcessedData() {
  const metadata = JSON.parse(fs.readFileSync(config.iconDataFile, "utf8"));
  const features = JSON.parse(fs.readFileSync(config.featuresFile, "utf8"));

  return { metadata, features };
}

export function getInputShape() {
  return [config.iconSize, config.iconSize, 1];
}

export function getFlattenedInputShape() {
  return [config.iconSize ** 2];
}

if (import.meta.main) {
  preprocessAllIcons().catch(console.error);
}
