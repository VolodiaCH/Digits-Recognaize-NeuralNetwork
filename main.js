const fs = require('fs');
const PNG = require('pngjs').PNG; // https://github.com/lukeapage/pngjs
const NeuralNetwork = require("./neural-network");

const softmax = vector => { // https://en.wikipedia.org/wiki/Softmax_function
    const sum = vector.reduce((acc, value) => acc + Math.exp(value), 0);
    return vector.map(value => Math.exp(value) / sum);
};

const getRandomInt = (min, max) => {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
}

class Main {
    constructor() {
        this.images = new Array();
    }

    readImgs = () => {
        let images = new Array();

        console.log("Parsing images in directory...\n");
        let t = new Date();

        // list all files in the directory
        fs.readdir("./train", (err, files) => {
            if (err) throw err;
            else {
                files.forEach((file, idx) => {
                    if (idx % 10000 === 0 && idx !== 0 || idx+1 === 60000) console.log(`${idx}/60000 images parsed`);

                    let img = new Array();

                    let data = fs.readFileSync(`./train/${file}`);
                    let png = PNG.sync.read(data);

                    for (let y = 0; y < png.height; y++) {
                        for (let x = 0; x < png.width; x++) {
                            let idx = (png.width * y + x) << 2;
                            let pixel = (png.data[idx] + png.data[idx + 1] + png.data[idx + 2]) > 1 ? 1 : 0;
        
                            img.push(pixel);
                        }
                    }

                    images.push({pixels: img, num: parseInt(file.slice(10, 11))});
                });
            }

            console.log(`\nAll images parsed successfuly; Completed in ${new Date()-t} ms\n`);

            this.images = images;
            this.createNN();
        });
    }

    createNN = () => {
        let t = new Date();
        console.log("Creating Neural Network architecture...");
        let inpN = 784;
        let outN = 10;
        const NN = new NeuralNetwork(inpN, [32, 16], outN);
        console.log(`Neural Network architecture successfuly created; Completed in ${new Date()-t} ms`);

        let inputs = this.images;
        let outputs = [];
        for (let y = 0; y < outN; y++) {
            let yArr = [];
            for (let x = 0; x < outN; x++) {
                if (y === x) yArr[x] = 1;
                else yArr[x] = 0;
            }
            outputs[y] = yArr;
        }

        t = new Date();
        console.log("Neural Network is starting to learn...");
        let error = 0;
        let correct = 0;

        let l = 60000;
        for (let i = 0; i < 3; i++) {
            console.log(`\nEpoch ${i+1}\n`);
            error = 0;
            correct = 0;

            for (let imgIdx = 0; imgIdx < l; imgIdx++) {
                let input = inputs[imgIdx].pixels;
                let output = outputs[inputs[imgIdx].num];
    
                let answer = NN.feedForward(input);
                answer = softmax(answer);
    
                let err = [];
                for (let i = 0; i < outN; i++) {
                    err[i] = answer[i] - output[i];
                }

                error += err.reduce((a, b) => a + b);
                let p = answer.indexOf(Math.max(...answer));
                let r = output.indexOf(1);
                if (p === r) correct++
    
                NN.backPropagation(err);

                if (imgIdx % 10000 === 0 && imgIdx !== 0 || imgIdx+1 === l) {
                    console.log(`${imgIdx}/60000 images parsed; Correct ${correct}/10000 (${(correct/10000*100).toFixed(2)}%)`);
                    correct = 0;
                }
            }

            NN.lambda /= 10;
        }
        
        console.log(`\nNeural Network learning finished; Completed in ${new Date() - t} ms\n`);
        console.log("Testing learn progress...");

        correct = 0;
        let tests = 100;
        for (let imgIdx = 0; imgIdx < tests; imgIdx++) {
            let input = inputs[imgIdx].pixels;
            let output = outputs[inputs[imgIdx].num];

            let answer = NN.feedForward(input);
            answer = softmax(answer)

            let err = [];
            for (let i = 0; i < outN; i++) {
                err[i] = answer[i] - output[i];
            }

            let p = answer.indexOf(Math.max(...answer));
            let r = output.indexOf(1);
            if (p === r) correct++;

            error += err.reduce((a, b) => Math.abs(a) + Math.abs(b));
        }

        // console.log(`Avarage error = ${error / 1000}`);
        console.log(`Correct: ${correct}/${tests} (${(correct/tests*100).toFixed(0)}%)`);
    }
}

const main = new Main();
main.readImgs();
// main.createNN();