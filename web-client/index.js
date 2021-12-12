const predict = async function(model, webcam){
//	image = await webcam.capture();
	image = tf.browser.fromPixels(webcamElement);
	image = image.expandDims(0).toFloat();
	console.log(image);
	image.print();
//	const prediction = await model.executeAsync(tfImg, ["StatefulPartitionedCall/sequential_1/sequential/random_crop/sub_1"]);
	const prediction = await model.executeAsync(image, ["StatefulPartitionedCall/sequential_1/sequential/random_crop/sub_1", 'StatefulPartitionedCall/sequential_1/dense_2/BiasAdd/ReadVariableOp', 'Identity:0']);
	data = await prediction[2];
	data.print();
	console.log(d);
	result = tf.softmax(data).argMax(-1);
	result.print();
}

const start = async function(){
	tf.setBackend('webgl');
	const model = await tf.loadGraphModel('/cs254a-final-project/demo/saved_models/CNN/model.json');
	webcamElement = document.getElementById('webcam');
	webcam = await tf.data.webcam(webcamElement);

	$(document).keydown(function(){
		predict(model, webcam);
//		screenShot = webcam.capture();
//		example = tf.fromPixels(screenShot);
//		webcamElement = document.getElementById('webcam');
//		tfImg = tf.browser.fromPixels(webcamElement);
//		tfImg = tfImg.expandDims(0);
//		const smalImg = tf.image.resizeBilinear(tfImg, [150, 150]);
//		const resized = tf.cast(smalImg, 'float32');
//		const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,150,150,3])
//		console.log(tfImg);
//		console.log(smalImg);
//		console.log(resized);
//		console.log(t4d);
//		smalImg.print();
//		predict(model, tfImg);
	});

}

$(document).ready(function(){
	start();
});
