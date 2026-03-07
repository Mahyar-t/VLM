/**
 * common.js - Handles shared UI components (sidebar, navigation, and transitions)
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Inject Sidebar
    const sidebarPlaceholder = document.getElementById('sidebar-placeholder');
    if (sidebarPlaceholder) {
        fetch('/html/sidebar.html')
            .then(response => response.text())
            .then(html => {
                sidebarPlaceholder.innerHTML = html;
                initializeSidebar();
                highlightActiveLink();
            })
            .catch(err => console.error('Failed to load sidebar:', err));
    } else {
        // 2. Initialize sidebar components
        initializeSidebar();
        highlightActiveLink();
    }
});

/**
 * Initialize sidebar toggle logic
 */
function initializeSidebar() {
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');

    if (!menuBtn || !sidebar || !overlay) return;

    function toggleSidebar() {
        menuBtn.classList.toggle('active');
        sidebar.classList.toggle('active');
        overlay.classList.toggle('active');
    }

    menuBtn.addEventListener('click', toggleSidebar);
    overlay.addEventListener('click', toggleSidebar);
}

/**
 * Highlight the active link in the sidebar and top-nav based on URL
 */
function highlightActiveLink() {
    const currentPath = window.location.pathname;
    const links = document.querySelectorAll('.sidebar-link, .top-nav-link');
    
    links.forEach(link => {
        const path = link.getAttribute('data-path') || link.getAttribute('href');
        if (currentPath === path || (currentPath === '/index.html' && path === '/') || (currentPath === '/' && path === '/index.html')) {
            link.classList.add('active');
            if (link.classList.contains('top-nav-link')) {
                link.style.color = 'var(--primary)';
                link.style.opacity = '1';
            }
        } else {
            link.classList.remove('active');
            if (link.classList.contains('top-nav-link')) {
                link.style.color = '';
                link.style.opacity = '';
            }
        }
    });
}

/**
 * Scroll-to-hide top bar logic
 */
(function () {
    let lastScroll = 0;
    window.addEventListener('scroll', function () {
        const topBar = document.querySelector('.top-bar');
        if (!topBar) return;

        const current = window.scrollY || document.documentElement.scrollTop;
        if (current > lastScroll && current > 80) {
            topBar.classList.add('hidden');
        } else {
            topBar.classList.remove('hidden');
        }
        lastScroll = current <= 0 ? 0 : current;
    }, { passive: true });
})();
