import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [emotion, setEmotion] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setEmotion(""); // Reset previous prediction
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select an audio file!");
            return;
        }
    
        const formData = new FormData();
        formData.append("audio", selectedFile);
        setLoading(true);
    
        try {
            const response = await axios.post("http://127.0.0.1:8000/api/predict/", formData);
            console.log("Response from backend:", response.data); // Debugging Line
    
            if (response.data.status === "success") { // ✅ Fix here
                setEmotion(response.data.emotion);
            } else {
                console.error("Error from backend:", response.data.message);
                alert("Error: " + response.data.message);
            }
        } catch (error) {
            console.error("Error predicting emotion:", error);
            alert("Error processing audio. Check console for details.");
        } finally {
            setLoading(false);
        }
    };
    


    return (
        <div className="container">
            <h1>🎤 Emotion Recognition</h1>
            <input type="file" onChange={handleFileChange} accept="audio/*" className="file-input" />
            <button onClick={handleUpload} className="upload-btn">
                {loading ? "Processing..." : "Predict Emotion"}
            </button>
            {emotion && <h2 className="result">Predicted Emotion: {emotion}</h2>}
        </div>
    );
};

export default App;
