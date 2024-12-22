import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class RoadDamageDetectionScreen extends StatefulWidget {
  @override
  _RoadDamageDetectionScreenState createState() =>
      _RoadDamageDetectionScreenState();
}

class _RoadDamageDetectionScreenState extends State<RoadDamageDetectionScreen> {
  Interpreter? _interpreter;
  String _result = "No prediction yet.";
  final picker = ImagePicker();
  img.Image? _selectedImage;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilenetv2.tflite');
      print("Model loaded successfully.");
    } catch (e) {
      print("Failed to load model: $e");
    }
  }

  Future<void> _pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      // Operasi async dilakukan di luar setState
      final imageBytes = await pickedFile.readAsBytes();
      final image = img.decodeImage(imageBytes);

      // setState hanya untuk memperbarui UI
      setState(() {
        _selectedImage = image;
      });

      // Jalankan model setelah UI diperbarui
      _runModel();
    }
  }

  Future<void> _runModel() async {
    if (_selectedImage == null || _interpreter == null) return;

    final input = _preprocessImage(_selectedImage!, 100, 100);
    final output = List.filled(2, 0.0).reshape([1, 2]);

    _interpreter!.run(input, output);

    setState(() {
      _result = "Predicted: ${_interpretOutput(output[0])}";
    });
  }

  Uint8List _preprocessImage(img.Image image, int targetWidth, int targetHeight) {
    final resizedImage = img.copyResize(image, width: targetWidth, height: targetHeight);

    // Pastikan image menjadi Float32List dan nilai pixel dinormalisasi
    final normalizedImage = Float32List(targetWidth * targetHeight * 3);
    int index = 0;

    // Mengambil nilai RGB dan menormalkan menjadi 0-1
    for (var y = 0; y < targetHeight; y++) {
      for (var x = 0; x < targetWidth; x++) {
        final pixel = resizedImage.getPixel(x, y);
        normalizedImage[index++] = (pixel.r / 255.0);  // Normalisasi merah
        normalizedImage[index++] = (pixel.g / 255.0); // Normalisasi hijau
        normalizedImage[index++] = (pixel.b / 255.0);  // Normalisasi biru
      }
    }

    return normalizedImage.buffer.asUint8List();
  }

  

  String _interpretOutput(List<double> output) {
    final labels = ['No Crack', 'Crack'];
    final maxIndex = output.indexWhere((value) => value == output.reduce((a, b) => a > b ? a : b));
    return labels[maxIndex];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Road Damage Detection")),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_selectedImage != null)
            Image.memory(Uint8List.fromList(img.encodePng(_selectedImage!))),
          SizedBox(height: 20),
          Text(
            _result,
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 20),
          ElevatedButton(
            onPressed: _pickImage,
            child: Text("Capture Image"),
          ),
        ],
      ),
    );
  }
}
