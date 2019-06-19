let fs = require('fs');
let Redis = require('ioredis');
let Jimp = require('jimp');
let Helpers = require("./Helpers");

let answerSet = [];

async function run(filenames) {
    let labels = JSON.parse(fs.readFileSync("imagenet_class_index.json"));
    let redis = new Redis({parser: 'javascript'});
    const model_filename = './models/mobilenet_v2_1.4_224_frozen.pb';
    const input_var = 'input';
    const output_var = 'MobilenetV2/Predictions/Reshape_1';
    const modelBuffer = fs.readFileSync(model_filename, {'flag': 'r'});

    console.log("Setting model");
    redis.call('AI.MODELSET', 'mobilenet', 'TF', 'CPU', 'INPUTS', input_var, 'OUTPUTS', output_var, modelBuffer);

    const image_height = 224;
    const image_width = 224;
    let p = 0;

    for (let i in filenames) {
        let filename = filenames[i];
        console.log("Reading image");
        let input_image = await Jimp.read(filename);

        let image = input_image.cover(image_width, image_height);
        let normalized = Helpers.normalizeRGB(image.bitmap.data, image.hasAlpha());

        let buffer = Buffer.from(normalized.buffer);

        console.log("Setting input tensor");
        redis.call('AI.TENSORSET', 'input_' + i,
            'FLOAT', 1, image_width, image_height, 3,
            'BLOB', buffer);

        console.log("Running model");
        redis.call('AI.MODELRUN', 'mobilenet', 'INPUTS', 'input_' + i, 'OUTPUTS', 'output_' + i);

        console.log("Getting output tensor");
        let out_data = await redis.callBuffer('AI.TENSORGET', 'output_' + i, 'BLOB');
        let out_array = Helpers.bufferToFloat32Array(out_data[out_data.length - 1]);

        let label = Helpers.argmax(out_array);

        answerSet.push({
            'filename': filename,
            'matches': labels[label - 1][1]
        });
        p++;
        if (p === filenames.length) {
            console.log("\n...OK I think I got something...\n");
            console.table(answerSet, ['filename', 'matches']);
        }
    }
}

let filenames = Array.from(process.argv);
filenames.splice(0, 2);

run(filenames);
