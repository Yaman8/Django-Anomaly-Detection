// Select elements here
const video = document.getElementById('video');
const videoControls = document.getElementById('video-controls');
const sideTab = document.getElementById('sideTab');
const playButton = document.getElementById('play');
const chapterButton = document.getElementById('chapter');
const playbackIcons = document.querySelectorAll('.playback-icons use');
const timeElapsed = document.getElementById('time-elapsed');
const duration = document.getElementById('duration');

const videoProgressGroup = document.getElementById('video-progress');
const extractProgressGroup = document.getElementById('extract-progress-initial');
const extractProgressFinal = document.getElementById('extract-progress-final');

const midControls = document.getElementById('controls-non-extract');
const midControlsExtract = document.getElementById('controls-extract');


const progressBar = document.getElementById('progress-bar');
const extractBarInitial = document.getElementById('extract-bar-initial')
const extractBarFinal = document.getElementById('extract-bar-final')

const seek = document.getElementById('seek');
const seekInitial = document.getElementById('seek-initial');
const seekFinal = document.getElementById('seek-final');

const seekTooltip = document.getElementById('seek-tooltip');
const seekTooltipInitial = document.getElementById('seek-tooltip-initial');
const seekTooltipFinal = document.getElementById('seek-tooltip-final');

// const volumeButton = document.getElementById('volume-button');
// const volumeIcons = document.querySelectorAll('.volume-button use');
// const volumeMute = document.querySelector('use[href="#volume-mute"]');
// const volumeLow = document.querySelector('use[href="#volume-low"]');
// const volumeHigh = document.querySelector('use[href="#volume-high"]');
// const volume = document.getElementById('volume');
const playbackAnimation = document.getElementById('playback-animation');
// const fullscreenButton = document.getElementById('fullscreen-button');
const videoContainer = document.getElementById('video-container');
// const fullscreenIcons = fullscreenButton.querySelectorAll('use');

const videoWorks = !!document.createElement('video').canPlayType;
if (videoWorks) {
    video.controls = false;
    videoControls.classList.remove('hidden');
}

// Add functions here

// togglePlay toggles the playback state of the video.
// If the video playback is paused or ended, the video is played
// otherwise, the video is paused
function togglePlay() {
    if (video.paused || video.ended) {
        video.play();
    } else {
        video.pause();
    }
}

// updatePlayButton updates the playback icon and tooltip
// depending on the playback state
function updatePlayButton() {
    playbackIcons.forEach((icon) => icon.classList.toggle('hidden'));

    if (video.paused) {
        playButton.setAttribute('data-title', 'Play (k)');
    } else {
        playButton.setAttribute('data-title', 'Pause (k)');
    }
}

// formatTime takes a time length in seconds and returns the time in
// minutes and seconds
function formatTime(timeInSeconds) {
    const result = new Date(timeInSeconds * 1000).toISOString().substring(11, 19);
    return {
        minutes: result.substring(3, 5),
        seconds: result.substring(6, 8),
    };
}

// initializeVideo sets the video duration, and maximum value of the
// progressBar
function initializeVideo() {
    const videoDuration = Math.round(video.duration);

    seek.setAttribute('max', videoDuration);
    seekInitial.setAttribute('max', videoDuration);
    seekFinal.setAttribute('max', videoDuration);

    progressBar.setAttribute('max', videoDuration);
    extractBarFinal.setAttribute('max', videoDuration);
    extractBarInitial.setAttribute('max', videoDuration);

    const time = formatTime(videoDuration);
    duration.innerText = `${time.minutes}:${time.seconds}`;
    duration.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`);
}

// updateTimeElapsed indicates how far through the video
// the current playback is by updating the timeElapsed element
function updateTimeElapsed() {
    const time = formatTime(Math.round(video.currentTime));
    timeElapsed.innerText = `${time.minutes}:${time.seconds}`;
    timeElapsed.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`);
    // console.log(video.currentTime);
    // console.log(time);
}

// updateProgress indicates how far through the video
// the current playback is by updating the progress bar
function updateProgress() {
    seek.value = Math.floor(video.currentTime);
    progressBar.value = Math.floor(video.currentTime);
}

// updateSeekTooltip uses the position of the mouse on the progress bar to
// roughly work out what point in the video the user will skip to if
// the progress bar is clicked at that point
function updateSeekTooltip(event) {
    const skipTo = Math.round(
        (event.offsetX / event.target.clientWidth) *
        parseInt(event.target.getAttribute('max'), 10)
    );
    seek.setAttribute('data-seek', skipTo);
    const t = formatTime(skipTo);
    seekTooltip.textContent = `${t.minutes}:${t.seconds}`;
    const rect2 = progressBar.getBoundingClientRect();
    seekTooltip.style.left = `${event.pageX - rect2.left}px`;
}

function updateSeekTooltipInitial(event) {
    const skipTo = Math.round(
        (event.offsetX / event.target.clientWidth) *
        parseInt(event.target.getAttribute('max'), 10)
    );
    seekInitial.setAttribute('data-seek', skipTo);
    const t = formatTime(skipTo);
    seekTooltipInitial.textContent = `${t.minutes}:${t.seconds}`;
    const rect = extractBarInitial.getBoundingClientRect();;
    seekTooltipInitial.style.left = `${event.pageX - rect.left}px`;
}

function updateSeekTooltipFinal(event) {
    const skipTo = Math.round(
        (event.offsetX / event.target.clientWidth) *
        parseInt(event.target.getAttribute('max'), 10)
    );
    seekFinal.setAttribute('data-seek', skipTo);
    const t = formatTime(skipTo);
    seekTooltipFinal.textContent = `${t.minutes}:${t.seconds}`;
    const rect = extractBarFinal.getBoundingClientRect();
    seekTooltipFinal.style.left = `${event.pageX - rect.left}px`;
}
// skipAhead jumps to a different point in the video when the progress bar
// is clicked
function skipAhead(event) {
    const skipTo = event.target.dataset.seek
        ? event.target.dataset.seek
        : event.target.value;
    video.currentTime = String(skipTo);
    progressBar.value = skipTo;
    seek.value = skipTo;
}
function skipAheadInitial(event) {
    const skipTo = event.target.dataset.seek
        ? event.target.dataset.seek
        : event.target.value;
    video.currentTime = String(skipTo);
    extractBarInitial.value = skipTo;
    seekInitial.value = skipTo;
}
function skipAheadFinal(event) {
    const skipTo = event.target.dataset.seek
        ? event.target.dataset.seek
        : event.target.value;
    video.currentTime = String(skipTo);
    extractBarFinal.value = skipTo;
    seekFinal.value = skipTo;
}

// updateVolume updates the video's volume
// and disables the muted state if active
// function updateVolume() {
//     if (video.muted) {
//         video.muted = false;
//     }

//     video.volume = volume.value;
// }

// updateVolumeIcon updates the volume icon so that it correctly reflects
// the volume of the video
// function updateVolumeIcon() {
//     volumeIcons.forEach((icon) => {
//         icon.classList.add('hidden');
//     });

//     volumeButton.setAttribute('data-title', 'Mute (m)');

//     if (video.muted || video.volume === 0) {
//         volumeMute.classList.remove('hidden');
//         volumeButton.setAttribute('data-title', 'Unmute (m)');
//     } else if (video.volume > 0 && video.volume <= 0.5) {
//         volumeLow.classList.remove('hidden');
//     } else {
//         volumeHigh.classList.remove('hidden');
//     }
// }

// toggleMute mutes or unmutes the video when executed
// When the video is unmuted, the volume is returned to the value
// it was set to before the video was muted
// function toggleMute() {
//     video.muted = !video.muted;

//     if (video.muted) {
//         volume.setAttribute('data-volume', volume.value);
//         volume.value = 0;
//     } else {
//         volume.value = volume.dataset.volume;
//     }
// }

// animatePlayback displays an animation when
// the video is played or paused
function animatePlayback() {
    playbackAnimation.animate(
        [
            {
                opacity: 1,
                transform: 'scale(1)',
            },
            {
                opacity: 0,
                transform: 'scale(1.3)',
            },
        ],
        {
            duration: 500,
        }
    );
}

// toggleFullScreen toggles the full screen state of the video
// If the browser is currently in fullscreen mode,
// then it should exit and vice versa.
function toggleFullScreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else if (document.webkitFullscreenElement) {
        // Need this to support Safari
        document.webkitExitFullscreen();
    } else if (videoContainer.webkitRequestFullscreen) {
        // Need this to support Safari
        videoContainer.webkitRequestFullscreen();
    } else {
        videoContainer.requestFullscreen();
    }
}

// updateFullscreenButton changes the icon of the full screen button
// and tooltip to reflect the current full screen state of the video
// function updateFullscreenButton() {
//     fullscreenIcons.forEach((icon) => icon.classList.toggle('hidden'));

//     if (document.fullscreenElement) {
//         fullscreenButton.setAttribute('data-title', 'Exit full screen (f)');
//     } else {
//         fullscreenButton.setAttribute('data-title', 'Full screen (f)');
//     }
// }

// hideControls hides the video controls when not in use
// if the video is paused, the controls must remain visible
function hideControls() {
    if (video.paused) {
        return;
    }

    videoControls.classList.add('hide');
}

// showControls displays the video controls
function showControls() {
    videoControls.classList.remove('hide');
}

//show the side bar
function hideSideTab() {
    sideTab.classList.add('hide');
}
function toggleSideTab() {
    sideTab.classList.toggle('hide');
    videoControls.classList.toggle('shrink');
}
function toggleExtract() {
    midControls.classList.toggle('hide');
    midControlsExtract.classList.toggle('hide');
    videoProgressGroup.classList.toggle('hide');
    extractProgressGroup.classList.toggle('hide');
    extractProgressFinal.classList.toggle('hide');
}


// Add eventlisteners here
playButton.addEventListener('click', togglePlay);
chapterButton.addEventListener('click', toggleSideTab);
video.addEventListener('play', updatePlayButton);
video.addEventListener('pause', updatePlayButton);
video.addEventListener('loadedmetadata', initializeVideo);
video.addEventListener('timeupdate', updateTimeElapsed);
video.addEventListener('timeupdate', updateProgress);
// video.addEventListener('volumechange', updateVolumeIcon);
video.addEventListener('click', togglePlay);
video.addEventListener('click', animatePlayback);
video.addEventListener('mouseenter', showControls);
video.addEventListener('mouseleave', hideControls);
videoControls.addEventListener('mouseenter', showControls);
videoControls.addEventListener('mouseleave', hideControls);

seek.addEventListener('mousemove', updateSeekTooltip);
seekInitial.addEventListener('mousemove', updateSeekTooltipInitial);
seekFinal.addEventListener('mousemove', updateSeekTooltipFinal);

seek.addEventListener('input', skipAhead);
seekInitial.addEventListener('input', skipAheadInitial);
seekFinal.addEventListener('input', skipAheadFinal);

// volume.addEventListener('input', updateVolume);
// volumeButton.addEventListener('click', toggleMute);
// fullscreenButton.addEventListener('click', toggleFullScreen);
// videoContainer.addEventListener('fullscreenchange', updateFullscreenButton);

// document.addEventListener('keyup', keyboardShortcuts);
