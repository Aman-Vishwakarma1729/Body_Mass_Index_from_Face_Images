captureBtn.addEventListener('click', () => {
  // ... (existing code to capture image)

  const canvas = document.getElementById('canvas');  // Update ID if needed
  const imageDataURL = canvas.toDataURL('image/jpeg');

  const formData = new FormData();
  formData.append('image', imageDataURL);

  fetch('/capture', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      if (data.success) {
          // Update the image display with the saved image filename
          const imagePreview = document.getElementById('capture');
          imagePreview.innerHTML = `<img src="/static/images/${data.filename}" alt="Captured Image">`;
      } else {
          console.error('Error capturing image:', data.error);
      }
  })
  .catch(error => console.error('Error:', error));
});
