import { config } from '@/lib/config'

export type CheckedVideo = {
	file: File
	url: string
	durationSec: number
	width: number
	height: number
}

export type VideoCheck = { ok: true; video: CheckedVideo } | { ok: false; message: string }

function readMetadata(url: string): Promise<{ duration: number; width: number; height: number }> {
	return new Promise((resolve, reject) => {
		const video = document.createElement('video')
		video.preload = 'metadata'
		video.muted = true
		video.onloadedmetadata = () =>
			resolve({ duration: video.duration, width: video.videoWidth, height: video.videoHeight })
		video.onerror = () => reject(new Error('unreadable'))
		video.src = url
	})
}

/** Same limits the API enforces; checking here saves a pointless upload. */
export async function checkVideoFile(file: File): Promise<VideoCheck> {
	const isMp4 = file.type === 'video/mp4' || /\.mp4$/i.test(file.name)
	if (!isMp4) {
		return { ok: false, message: `"${file.name}" is not an MP4 file.` }
	}

	const maxBytes = config.maxUploadMb * 1024 * 1024
	if (file.size > maxBytes) {
		return {
			ok: false,
			message: `The file is ${(file.size / 1024 / 1024).toFixed(0)} MB; the demo accepts up to ${config.maxUploadMb} MB.`,
		}
	}

	const url = URL.createObjectURL(file)

	try {
		const meta = await readMetadata(url)

		if (!Number.isFinite(meta.duration) || meta.duration <= 0) {
			URL.revokeObjectURL(url)
			return { ok: false, message: 'The video length could not be read.' }
		}

		if (meta.duration > config.maxDurationSec + 0.5) {
			URL.revokeObjectURL(url)
			return {
				ok: false,
				message: `The video is ${meta.duration.toFixed(0)} s long; the demo accepts up to ${config.maxDurationSec} s. Trim it and try again.`,
			}
		}

		return {
			ok: true,
			video: { file, url, durationSec: meta.duration, width: meta.width, height: meta.height },
		}
	} catch {
		URL.revokeObjectURL(url)
		// Some browsers cannot decode HEVC MP4s; the server may still, but we cannot check length.
		return {
			ok: false,
			message: 'This browser cannot read the video. Re-encode it as H.264 MP4 and try again.',
		}
	}
}
