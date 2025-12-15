async function analyzeSentiment() {
    const text = document.getElementById('inputText').value;
    const btn = document.getElementById('analyzeBtn');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const sentimentLabel = document.getElementById('sentimentLabel');
    const sentimentBox = document.querySelector('.sentiment-box');
    const confidenceValue = document.getElementById('confidenceValue');

    if (!text.trim()) {
        return;
    }

    // Reset UI
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');

    try {
        // Use relative path if served from same origin, or full URL if separate
        // For this setup, we'll assume same origin or configured proxy
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Update UI with result
        sentimentLabel.textContent = data.sentiment;
        confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        sentimentBox.className = 'sentiment-box'; // Reset classes
        if (data.sentiment === 'Positive') {
            sentimentBox.classList.add('positive');
        } else {
            sentimentBox.classList.add('negative');
        }
        
        resultDiv.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        errorDiv.textContent = 'An error occurred while analyzing sentiment. Please try again.';
        errorDiv.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Sentiment';
    }
}
