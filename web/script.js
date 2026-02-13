
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

// Basic implementation of the client-side logic described in export_onnx.py

const STATUS = document.getElementById('status');

async function loadNpz(url) {
    STATUS.textContent = "Downloading gaussians.npz...";
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    // We need a way to parse .npz (which is a zip file)
    // Using JSZip (loaded in index.html)
    const zip = await JSZip.loadAsync(buffer);

    const data = {};
    for (const [filename, file] of Object.entries(zip.files)) {
        if (!filename.endsWith('.npy')) continue;
        const key = filename.replace('.npy', '');
        const arrayBuffer = await file.async('arraybuffer');
        data[key] = parseNpym(arrayBuffer);
    }
    return data;
}

// Simple .npy parser (for float16/float32/int32/uint16/uint8)
function parseNpym(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer.slice(1, 6)));
    if (magic !== 'NUMPY') throw new Error('Not a .npy file');

    const versionMajor = view.getUint8(6);
    const headerLen = versionMajor === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
    const offset = versionMajor === 1 ? 10 : 12;

    const headerStr = new TextDecoder().decode(buffer.slice(offset, offset + headerLen));
    const header = eval(`(${headerStr.toLowerCase().replace('(', '[').replace(')', ']')})`); // Safety warning: eval

    const dtype = header.descr;
    const shape = header.shape;
    const dataOffset = offset + headerLen;
    // Align to whatever specific byte alignment might be needed or just slice

    const dataBuffer = buffer.slice(dataOffset);
    let array;

    if (dtype === '<f4') array = new Float32Array(dataBuffer);
    else if (dtype === '<f2') {
        // Javascript doesn't support Float16Array natively across all browsers yet, 
        // but we can treat as Uint16 and convert on fly, or use a library.
        // For this demo, let's assume we convert to Float32 immediately.
        array = decodeFloat16(new Uint16Array(dataBuffer));
    }
    else if (dtype === '<i4') array = new Int32Array(dataBuffer);
    else if (dtype === '<u2') array = new Uint16Array(dataBuffer);
    else if (dtype === '|u1') array = new Uint8Array(dataBuffer);
    else throw new Error(`Unsupported dtype: ${dtype}`);

    // Reshape if needed (simple flat array returned here)
    // Implementing full reshape is complex, we'll return flat + shape info
    array.shape = shape;
    return array;
}

// Float16 decode (approximate)
function decodeFloat16(uint16Array) {
    const float32Array = new Float32Array(uint16Array.length);
    for (let i = 0; i < uint16Array.length; i++) {
        const h = uint16Array[i];
        const s = (h >> 15) & 0x1;
        const e = (h >> 10) & 0x1f;
        const f = h & 0x3ff;
        if (e === 0) {
            float32Array[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        } else if (e === 0x1f) {
            float32Array[i] = f === 0 ? (s ? -Infinity : Infinity) : NaN;
        } else {
            float32Array[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
        }
    }
    return float32Array;
}

async function main() {
    try {
        // 1. Load Gaussians Data
        const svqData = await loadNpz('../test_export_out/gaussians.npz'); // pointing to test output for demo
        console.log("SVQ Data loaded:", svqData);
        STATUS.textContent = "Data loaded. Reconstructing...";

        // 2. Reconstruct Attributes (Simplified for Demo)
        // In a real app, you'd implement the gather logic here:
        // scale = scale_codebook[scale_index]
        // For this demo, we'll just create dummy buffers if the gathered ones are complex to build in JS without a library like tensorflow.js or onnxruntime for tensor ops.
        // Actually, onnxruntime *is* available.
        // But the reconstruction happens *before* the model inference usually, or we pass indices?
        // The export_onnx.py says:
        // "Client-side reconstruction is a single gather per attribute"
        // "Inputs: xyz, features_static, features_view, t_norm"

        // Let's assume we have N points.
        const N = svqData['xyz'].shape[0];

        // Mock reconstruction for the demo since we don't have the Gather impl in JS readily written out
        // In production, you'd map indices to codebooks.
        const featuresStatic = new Float32Array(N * 3); // Random dummy
        const featuresView = new Float32Array(N * 3);   // Random dummy

        const xyz = svqData['xyz']; // Alread f32 (converted from f16 in parse)

        // 3. Initialize Session
        STATUS.textContent = "Loading ONNX Model...";
        const session = await ort.InferenceSession.create('../test_export_out/mlp_inference.onnx');
        STATUS.textContent = "Model loaded. Running Inference...";

        // 4. Run Inference
        const tVal = 0.5;

        const feeds = {
            xyz: new ort.Tensor('float32', xyz, [N, 3]),
            features_static: new ort.Tensor('float32', featuresStatic, [N, 3]),
            features_view: new ort.Tensor('float32', featuresView, [N, 3]),
            t_norm: new ort.Tensor('float32', [tVal], [1])
        };

        const results = await session.run(feeds);
        console.log("Inference Results:", results);

        const opacity = results.opacity.data;
        const sh_rest = results.sh_rest.data;
        const dc = results.dc.data;

        STATUS.textContent = `Inference Success! Computed ${opacity.length} opacities.`;

        // Control hook
        document.getElementById('timeSlider').addEventListener('input', async (e) => {
            const t = parseFloat(e.target.value);
            document.getElementById('timeValue').textContent = t.toFixed(2);
            // Re-run inference (in a real app, this would be in the render loop)
            const newFeeds = { ...feeds, t_norm: new ort.Tensor('float32', [t], [1]) };
            const newResults = await session.run(newFeeds);
            console.log("Updated result for t=", t);
        });

    } catch (e) {
        console.error(e);
        STATUS.textContent = "Error: " + e.message;
    }
}

main();
