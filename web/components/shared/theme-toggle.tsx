'use client'

import { MoonIcon, SunIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'

export const themeInitScript = `(function(){try{var t=localStorage.getItem('theme');var d=t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.classList.toggle('dark',d)}catch(e){}})()`

export function ThemeToggle() {
	function toggle() {
		const dark = !document.documentElement.classList.contains('dark')
		document.documentElement.classList.toggle('dark', dark)
		try {
			localStorage.setItem('theme', dark ? 'dark' : 'light')
		} catch {
			// Storage can be blocked; the toggle still works for this page view.
		}
	}

	return (
		<Button variant='ghost' size='icon' onClick={toggle} aria-label='Toggle dark mode'>
			<SunIcon className='hidden dark:block' />
			<MoonIcon className='dark:hidden' />
		</Button>
	)
}
