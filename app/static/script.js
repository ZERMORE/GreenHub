function runModel() {
    const lane1 = document.getElementById('lane1').value;
    const lane2 = document.getElementById('lane2').value;
    const lane3 = document.getElementById('lane3').value;

    const data = { lane1, lane2, lane3 };

    fetch('/run-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('modelOutput').innerText = result.output;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}