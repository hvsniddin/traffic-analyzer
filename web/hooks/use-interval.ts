'use client'

import * as React from 'react'

export function useInterval(callback: () => void, delayMs: number | null) {
	const callbackRef = React.useRef(callback)

	React.useEffect(() => {
		callbackRef.current = callback
	})

	React.useEffect(() => {
		if (delayMs === null) return

		const id = setInterval(() => callbackRef.current(), delayMs)
		return () => clearInterval(id)
	}, [delayMs])
}
