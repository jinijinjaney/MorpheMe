document.addEventListener("DOMContentLoaded", function () {
    let seeResultsBtn = document.getElementById("see-results-btn");
    let resultContainer = document.getElementById("result-container");
    let resultText = document.getElementById("result-text");
    let resultImg = document.getElementById("result-img");
    let isCaptured = false; // Flag to track capture vs upload
    let downloadContainer = document.querySelector('.download-container');

    // Hide elements initially
    seeResultsBtn.style.display = "none";
    resultContainer.style.display = "none";
    resultText.style.display = "none";
    resultImg.style.display = "none";

    // Update these functions
function showSeeResultsButton() {
    document.getElementById("see-results-btn").style.display = "inline-block";
}

function hideSeeResultsButton() {
    document.getElementById("see-results-btn").style.display = "none";
}

function resetResultDisplay() {
    let resultContainer = document.getElementById("result-container");
    let resultText = document.getElementById("result-text");
    let resultImg = document.getElementById("result-img");
    
    resultContainer.style.display = "none";
    resultText.style.display = "none";
    resultImg.style.display = "none";
    resultImg.classList.remove("visible"); // Remove animation class
    hideDownloadButton(); // Hide download button too
}

// Function to show download button
function showDownloadButton() {
    downloadContainer.style.display = "block";
}

// Function to hide download button
function hideDownloadButton() {
    downloadContainer.style.display = "none";
}

    // ✅ Show swapped image only when SEE RESULTS is clicked
    seeResultsBtn.addEventListener("click", function () {
        if (resultImg.src && resultImg.src !== "") {
            resultContainer.style.display = "block";  
            resultText.style.display = "block";  
            resultImg.style.display = "block";  
        } else {
            alert("No swapped image available! Please capture or upload an image first.");
        }
    });

    // Navbar background change on scroll
    window.addEventListener("scroll", function () {
        const navbar = document.querySelector(".navbar");
        if (window.scrollY > 50) {
            navbar.classList.add("bg-dark", "shadow");
        } else {
            navbar.classList.remove("shadow");
        }
    });

    // 📤 Handle Image Upload
    document.getElementById("upload-btn").addEventListener("click", function () {
        let fileInput = document.getElementById("imageInput");
        let templateSelect = document.getElementById("templateSelect").value;

        if (fileInput.files.length === 0) {
            alert("Please select an image!");
            return;
        }

        let formData = new FormData();
        formData.append("image", fileInput.files[0]);
        formData.append("template", templateSelect);

        // Hide result image until SEE RESULTS is clicked again
        resetResultDisplay();
        hideSeeResultsButton();

        fetch("http://127.0.0.1:5000/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            let imgUrl = URL.createObjectURL(blob);
            document.getElementById("result-img").src = imgUrl;
            isCaptured = false;
            showSeeResultsButton(); 
        })
        .catch(error => console.error("Error:", error));
    });

    // Custom file upload button (Prevents clicking "No file chosen")
    document.getElementById("choose-file-btn").addEventListener("click", function () {
        document.getElementById("imageInput").click(); // Triggers the file input
    });

    document.getElementById("imageInput").addEventListener("change", function () {
        let fileName = this.files.length > 0 ? this.files[0].name : "No file chosen";
        document.getElementById("file-name").textContent = fileName;
    });

    // Webcam capture functionality
    let video = document.getElementById("video");

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error("Webcam access denied!", error);
    });

    // 📸 Handle Webcam Capture
    document.getElementById("capture-btn").addEventListener("click", function () {
        let templateSelect = document.getElementById("templateSelect").value;
        if (!templateSelect) {
            alert("Please select a template!");
            return;
        }

        let formData = new FormData();
        formData.append("template", templateSelect);

        // Hide previous results until SEE RESULTS is clicked again
        resetResultDisplay();
        hideSeeResultsButton();

        fetch("/capture", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => { throw new Error("Failed to capture image: " + text); });
            }
            return response.blob();
        })
        .then(blob => {
            let imgUrl = URL.createObjectURL(blob);
            document.getElementById("result-img").src = imgUrl;
            showSuccessPopup("Image Captured & Saved Successfully!");
            isCaptured = true;
            showSeeResultsButton();
        })
        .catch(error => {
            console.error("Error:", error);
            alert(error.message);
        });
    });

    // 🎉 Function to show success popup
    function showSuccessPopup(message) {
        let popup = document.createElement("div");
        popup.classList.add("success-popup");
        popup.innerHTML = `
            <div class="popup-content">
                <i class="fas fa-check-circle"></i>
                <p>${message}</p>
            </div>
        `;
    
        document.body.appendChild(popup);
    
        // Fade in
        setTimeout(() => {
            popup.style.opacity = "1";
            popup.style.transform = "translate(-50%, -50%) scale(1)";
        }, 50);
    
        // Fade out after 3 seconds
        setTimeout(() => {
            popup.style.opacity = "0";
            popup.style.transform = "translate(-50%, -50%) scale(0.9)";
            setTimeout(() => popup.remove(), 500);
        }, 3000);
    }

// Update the see-results-btn click handler to show download button
document.getElementById("see-results-btn").addEventListener("click", function () {
    let resultContainer = document.getElementById("result-container");
    let resultText = document.getElementById("result-text");
    let resultImg = document.getElementById("result-img");

    // Ensure an image exists before displaying
    if (resultImg.src && resultImg.src !== "") {
        resultContainer.style.display = "flex";  
        resultText.style.display = "block";  
        resultImg.style.display = "block";
        
        // Add animation class
        resultImg.classList.add("visible");
        
        // Show download button
        showDownloadButton();
        
        // Scroll to the result container
        resultContainer.scrollIntoView({ behavior: "smooth" });
    } else {
        alert("No swapped image available! Please capture or upload an image first.");
    }
});
    
    // 🔥 Hover Effects on "SEE RESULT" Button
    seeResultsBtn.addEventListener("mouseover", function() {
        this.style.backgroundColor = "#e04e2a";
        this.style.transform = "scale(1.05)";
    });

    seeResultsBtn.addEventListener("mouseout", function() {
        this.style.backgroundColor = "#ff5733";
        this.style.transform = "scale(1)";
    });

    // Add download functionality
document.getElementById("download-btn").addEventListener("click", function() {
    let resultImg = document.getElementById("result-img");
    
    // Create a link element
    let downloadLink = document.createElement("a");
    downloadLink.href = resultImg.src;
    downloadLink.download = "morpheme-image.png"; // Name of the downloaded file
    
    // Append to body, click it, then remove it
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    
    // Show success message
    showSuccessPopup("Image downloaded successfully!");
});
});