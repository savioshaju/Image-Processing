// Simple JavaScript for additional interactivity
document.addEventListener('DOMContentLoaded', function() {
    // File input label text update
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (this.files.length > 0) {
                label.textContent = this.files[0].name;
            } else {
                label.textContent = 'Choose an image';
            }
        });
    }
    
    // Add fade-in animation to elements
    const animateElements = document.querySelectorAll('.feature, .image-box, .text-results');
    animateElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    });
    
    // Trigger animation
    setTimeout(() => {
        animateElements.forEach((element, index) => {
            setTimeout(() => {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }, 100);
});