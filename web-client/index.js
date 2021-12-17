const predict = async function(model, webcam){
//	image = tf.browser.fromPixels(webcamElement);
	image = await webcam.capture();
	image = image.expandDims(0).toFloat();
	const prediction = model.predict(image);
	score = tf.softmax(prediction);
	result = await score.argMax(-1).data();
	confidence = await score.max().data();
	out = {score:score, result:result[0], confidence:confidence[0]};
	return out;
}

const changeText = function(backgroundColor, textColor, text){
	$("#text").css("background-color", backgroundColor,);
        $("#text").css("color", textColor);
        $("#text").html(text);
}

const predictAndChangeTextRecursive = async function(model, webcam, prediction, counter){
	console.log(prediction);
	if(prediction.confidence >= 0.95){
		document.getElementById('webcam').pause();
		if(prediction.result == 0){
			changeText("black", "green", "Access Granted");
		}
		else{
			changeText("black", "red", "Access Denied");
		}
	}
	else{
		if(counter >= 45){
                        changeText("black", "white", "Scanning...<br>Please Move Around to Get Better Images");
                }
		prediction = await predict(model, webcam);
		predictAndChangeTextRecursive(model, webcam, prediction, counter+1);
	}
}

const predictAndChangeText = async function(model, webcam){
	changeText("black", "white", "Scanning...");
	var prediction = await predict(model, webcam);
	predictAndChangeTextRecursive(model, webcam, prediction, 1);
	/*let counter = 1;
	while(prediction.confidence < 0.95 && counter <= 30){
		console.log(counter);
		prediction = await predict(model, webcamElement);
		console.log(counter);
		console.log(prediction);
		counter++;
		if(counter >= 45){
			changeText("black", "white", "Scanning...<br>Please Move Around to Get Better Images");
		}
	}
	webcamElement.pause();
	if(prediction.result == 0){
		changeText("black", "green", "Access Granted");
	}
	else{
		changeText("black", "red", "Access Denied");
	}*/

}

const start = async function(){
	$("#text").text("Loading Saved Model...");
	const model = await tf.loadGraphModel('/cs254a-final-project/demo/saved_models/graphs/CNN/model.json');
//	const model = await tf.loadLayersModel('/cs254a-final-project/demo/saved_models/layers/CNN/model.json');
	$("#text").text("Loading Webcam...");
	webcamElement = document.getElementById('webcam');
	webcam = await tf.data.webcam(webcamElement);
	changeText("black", "white", "Tap or Click to Start Scanning");
	const scan = function(){
		if(webcamElement.paused){
			webcamElement.play();
			changeText("transparent", "black", "");
		}
		else{
			predictAndChangeText(model, webcam);
		}
	}

	$(document).keydown(function(e){
		if(e.which == 32){
			scan();
		}
	});
	$("video").on('click touchstart', scan);
}

$(document).ready(start);
