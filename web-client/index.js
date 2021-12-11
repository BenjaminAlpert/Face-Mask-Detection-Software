
const start = async function(){
	const model = await tf.loadLayersModel('/cs254a-final-project/demo/saved_models/CNN/model.json');
}

$(document).ready(function(){

	start();
	webcam = tf.data.webcam(document.getElementById('webcam'));

	$(document).keydown(function(){
//		screenShot = webcam.capture();
//		example = tf.fromPixels(screenShot);
		webcamElement = document.getElementById('webcam');
		example = tf.fromPixels(webcamElement);
		prediction = model.predict(example);
		console.log(prediction);
	});
});
