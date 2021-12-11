// Define a model for linear regression.

$(document).ready(function(){

	const model = tf.loadLayersModel('file:///home/benjamin/workspace/academics-uvm-f21-cs254a-final-project/javascript/saved_models/CNN/model.json');

	webcam = tf.data.webcam(document.getElementById('webcam'));

	$(document).keydown(function(){
		screenShot = webcam.capture();
		example = tf.fromPixels(screenShot);
		prediction = model.predict(example);
		console.log(prediction);
	});
});
