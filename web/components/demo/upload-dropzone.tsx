'use client'

import { FileVideoIcon, UploadIcon } from 'lucide-react'
import * as React from 'react'

import { config } from '@/lib/config'
import { cn } from '@/lib/utils'

type UploadDropzoneProps = {
	disabled?: boolean
	onFile: (file: File) => void
}

export function UploadDropzone({ disabled, onFile }: UploadDropzoneProps) {
	const inputRef = React.useRef<HTMLInputElement>(null)
	const [dragging, setDragging] = React.useState(false)

	function handleFiles(files: FileList | null) {
		const file = files?.[0]
		if (file) onFile(file)
	}

	return (
		<label
			onDragOver={(event) => {
				event.preventDefault()
				if (!disabled) setDragging(true)
			}}
			onDragLeave={() => setDragging(false)}
			onDrop={(event) => {
				event.preventDefault()
				setDragging(false)
				if (!disabled) handleFiles(event.dataTransfer.files)
			}}
			className={cn(
				'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-colors',
				dragging ? 'border-primary bg-primary/5' : 'hover:border-foreground/30 hover:bg-muted/50',
				disabled && 'pointer-events-none opacity-50',
			)}
		>
			<span className='flex size-12 items-center justify-center rounded-full bg-muted'>
				{dragging ? <FileVideoIcon className='size-5' /> : <UploadIcon className='size-5' />}
			</span>
			<span className='grid gap-1'>
				<span className='font-medium'>Drop an MP4 here or choose a file</span>
				<span className='text-sm text-muted-foreground'>
					Up to {config.maxDurationSec / 60 >= 1 ? `${config.maxDurationSec / 60} min` : `${config.maxDurationSec} s`} and{' '}
					{config.maxUploadMb} MB. Best results come from this intersection camera at any resolution.
				</span>
			</span>
			<input
				ref={inputRef}
				type='file'
				accept='video/mp4,.mp4'
				className='sr-only'
				disabled={disabled}
				onChange={(event) => {
					handleFiles(event.target.files)
					// Allow picking the same file again after an error.
					event.target.value = ''
				}}
			/>
		</label>
	)
}
