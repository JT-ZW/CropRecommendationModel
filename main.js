document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        N: document.getElementById('N').value,
        P: document.getElementById('P').value,
        K: document.getElementById('K').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        ph: document.getElementById('ph').value,
        rainfall: document.getElementById('rainfall').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.status === 'success') {
            document.getElementById('result').classList.remove('hidden');
            document.getElementById('cropName').textContent = data.crop.toUpperCase();
            document.getElementById('confidence').textContent = 
                `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

            // Display alternative crops
            const alternativeCropsList = document.getElementById('alternativeCropsList');
            alternativeCropsList.innerHTML = ''; // Clear previous list

            data.alternative_crops.forEach(crop => {
                const listItem = document.createElement('li');
                listItem.innerHTML = `<span>${crop.crop.toUpperCase()}</span> - Confidence: <span>${(crop.confidence * 100).toFixed(2)}%</span>`;
                alternativeCropsList.appendChild(listItem);
            });

        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
});