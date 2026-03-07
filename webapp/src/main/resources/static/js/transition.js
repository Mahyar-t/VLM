/**
 * transition.js - Handles smooth page transitions (fade-in/fade-out)
 */

(function () {
    // 1. Initial state: body hidden (if not already set in CSS/HTML for a flash-free experience)
    // We do this via JS to ensure people without JS still see the content immediately.
    if (document.body.style.opacity === "") {
        document.body.style.opacity = "0";
    }

    // 2. Fade-in on load
    window.addEventListener("load", function () {
        requestAnimationFrame(() => {
            document.body.style.transition = "opacity 0.6s ease-in-out";
            document.body.style.opacity = "1";
        });
    });

    /**
     * Handle the fade-out transition and navigate
     * @param {string} href - The target URL
     */
    function performTransition(href) {
        document.body.style.opacity = "0";
        setTimeout(() => {
            window.location.href = href;
        }, 600);
    }

    // 3. Global link listener for transitions
    document.addEventListener("click", function (e) {
        const link = e.target.closest("a");
        if (!link) return;

        const href = link.getAttribute("href");
        const target = link.getAttribute("target");

        // Only transition for internal links that aren't anchors and don't open in new tabs
        if (
            href &&
            !href.startsWith("http") &&
            !href.startsWith("#") &&
            !href.startsWith("javascript:") &&
            target !== "_blank" &&
            href !== window.location.pathname &&
            href !== window.location.pathname + window.location.search
        ) {
            // Check if we should ignore this link (e.g. for specific functional buttons)
            if (link.classList.contains("no-transition")) return;

            e.preventDefault();

            // Close sidebar if it exists and is active (reusing IDs from common.js)
            const sidebar = document.getElementById("sidebar");
            const overlay = document.getElementById("overlay");
            const menuBtn = document.getElementById("menuBtn");

            if (sidebar && sidebar.classList.contains("active")) {
                sidebar.classList.remove("active");
                if (overlay) overlay.classList.remove("active");
                if (menuBtn) menuBtn.classList.remove("active");
                
                // Slight delay to allow sidebar to start closing before fade
                setTimeout(() => performTransition(href), 150);
            } else {
                performTransition(href);
            }
        }
    });

    // Handle browser back/forward buttons (ensure page isn't stuck at opacity 0)
    window.addEventListener("pageshow", function (event) {
        if (event.persisted) {
            document.body.style.opacity = "1";
        }
    });
})();
