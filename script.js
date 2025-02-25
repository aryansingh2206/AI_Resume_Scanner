document.getElementById("resumeForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const resumeFile = document.getElementById("resume").files[0];
    const jobDescription = document.getElementById("jobDescription").value;
    const resultDiv = document.getElementById("result");
    const loader = document.getElementById("loader");

    if (!resumeFile || !jobDescription) {
        alert("Please upload a resume and enter a job description.");
        return;
    }

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescription);

    // Show loader and clear previous results
    resultDiv.innerText = "";
    loader.style.display = "block";

    try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const result = await response.json();

        if (!result || !("matching_percentage" in result) || !("classification" in result)) {
            throw new Error("Invalid response from server.");
        }

        // Display match percentage and classification
        resultDiv.innerHTML = `
            <strong>Classification:</strong> ${result.classification} <br>
            <strong>Match Score:</strong> ${result.matching_percentage}%
        `;
        resultDiv.style.color = result.matching_percentage > 50 ? "#28a745" : "#dc3545"; // Green if > 50%, Red otherwise

    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerText = `Error: ${error.message}`;
        resultDiv.style.color = "#dc3545";
    } finally {
        loader.style.display = "none";
    }
});
