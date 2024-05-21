const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const searchInput = document.querySelector('.search');

uploadButton.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const selectedFile = fileInput.files[0];
  if (selectedFile) {
    const formData = new FormData();
    formData.append('filename', selectedFile);

    fetch('/main', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        if (data.prediction) {
          const flashMessage = document.createElement('div');
          flashMessage.classList.add('flash-message');
          flashMessage.textContent = `Predicted Genre: ${data.prediction}`;
          document.body.appendChild(flashMessage);

          displayRecommendations(data.recommendations,data.header); // Call function to display recommendations
          
          // Redirect after 5 seconds
          setTimeout(() => {
            window.location.href = "/main"; 
          }, 50000);
        } else {
          console.error('Error:', data.error);
        }
      })
      .catch(error => {
        console.error('Error uploading file:', error);
      });
  }
});
searchInput.addEventListener('keyup', (event) => {
  if (event.key === 'Enter') {
    const searchTerm = event.target.value;
    
    // Make sure the search term isn't empty
    if (searchTerm) {

      const formData = new FormData();
      formData.append('search_term', searchTerm); // Send search term as form data
      
      fetch('/main', { // Fetch from /main route
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok.');
        }
        return response.json(); 
      }) 
      .then(data => {

        if (data.recommendations && data.recommendations.length > 0) {
          displayRecommendations(data.recommendations,data.header);
          displayFlashMessage(`Showing results for: ${searchTerm}`, "success");
        } else {
         displayRecommendations(data.recommendations,data.header);
        }
      })
      .catch(error => {
        displayFlashMessage("Error fetching recommendations. Please try again.", "error");
      });
    } else {
      recommendationsDiv.style.display = 'none'; // Hide if search term is empty
    }
  }
});
document.addEventListener('DOMContentLoaded', () => {
  const realtimeButton = document.getElementById('realtimebutton');
  const resultElement = document.getElementById('flash-message');

  let mediaRecorder;

  realtimeButton.addEventListener('click', () => {
    resultElement.textContent = "Predicting...";
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

          const formData = new FormData();
          formData.append('audio_file', audioBlob);

          fetch('/predict_realtime', {
            method: 'POST',
            body: formData
          })
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok.')
            }
            return response.json();
          })
          .then(data => {
            if (data.genre) {
              resultElement.textContent = "Genre: " + data.genre;
              resultElement.classList.add("flash-message");
              displayRecommendations(data.recommendations,data.header); // Call function to display recommendations
              setTimeout(() => {
                resultElement.textContent = "";
                resultElement.classList.remove("flash-message");
              },8000);
            } else {
              resultElement.textContent = "Error: No genre prediction available.";
            }
          })
          .catch(error => {
            console.error('Error:', error);
            resultElement.textContent = "Error: " + error;
          });
        };

        mediaRecorder.start();

        setTimeout(() => {
          mediaRecorder.stop();
        }, 10000);
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
        resultElement.textContent = "Error: Could not access microphone.";
      });
  });
});

function displayRecommendations(recommendations,text) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = ''; // Clear previous recommendations
    if (recommendations && recommendations.length > 0) {
        const h2 = document.createElement('h2'); // Create <h2> element
        h2.textContent = text;       // Set the text content
        recommendationsDiv.appendChild(h2);      // Add the <h2> to the div

        const ul = document.createElement('ul');
        recommendations.forEach(recommendation => {      const li = document.createElement('li');
      li.textContent = `${recommendation.song_name} by ${recommendation.artist_name}`;
      const link = document.createElement('a');
      link.href = recommendation.link;
      link.textContent = 'Listen';
      li.appendChild(link);
      ul.appendChild(li);
    });
    recommendationsDiv.appendChild(ul);
    recommendationsDiv.style.display = 'block';
  }
else {
        recommendationsDiv.innerHTML = "<p>No recommendations found for this genre.</p>";
        recommendationsDiv.style.display = 'block'; // Show the div even when no recommendations are found
    }
}
