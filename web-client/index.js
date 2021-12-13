const predict = async function(model, webcamElement){
	image = tf.browser.fromPixels(webcamElement);
	image = image.expandDims(0).toFloat();
	const prediction = model.predict(image);
	score = tf.softmax(prediction);
	result = await score.argMax(-1).data();
	confidence = await score.max().data();
	out = {score:score, result:result[0], confidence:confidence[0]};
	console.log(out);
	return out;
}


const predictAndChangeText = async function(model, webcamElement){
	$("#text").css("background-color", "black");
	$("#text").css("color", "white");
	$("#text").text("Scanning...");
	prediction = await predict(model, webcamElement);
	while(prediction.confidence < 0.95){
		prediction = await predict(model, webcamElement);
	}
	webcamElement.pause();
	if(prediction.result == 0){
		$("#text").css("background-color", "black");
		$("#text").css("color", "green");
		$("#text").text("Access Granted");
	}
	else{
		$("#text").css("background-color", "black");
		$("#text").css("color", "red");
		$("#text").text("Access Denied");
	}

}

const start = async function(){
	$("#text").text("Loading Saved Model...");
	const model = await tf.loadGraphModel('/cs254a-final-project/demo/saved_models/graphs/CNN/model.json');
//	const model = await tf.loadLayersModel('/cs254a-final-project/demo/saved_models/layers/CNN/model.json');
	$("#text").text("Loading Webcam...");
	webcamElement = document.getElementById('webcam');
	webcam = await tf.data.webcam(webcamElement);
	$("#text").text("");

	$(document).keydown(function(e){
		if(e.which == 32){
			if(webcamElement.paused){
				webcamElement.play();
				$("#text").css("background-color", "transparent");
				$("#text").text("");
			}
			else{
				//predict(model, webcamElement).then(changeText);
				predictAndChangeText(model, webcamElement);
			}
		}
	});

}

$(document).ready(function(){
	start();
});
