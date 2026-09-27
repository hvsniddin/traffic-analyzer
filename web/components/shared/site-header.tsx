'use client'

import { MenuIcon, XIcon } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import * as React from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

import { ThemeToggle } from './theme-toggle'

export const navItems = [
	{ href: '/demo/', label: 'Live demo' },
	{ href: '/samples/', label: 'Results' },
	{ href: '/eda/', label: 'EDA' },
	{ href: '/approach/', label: 'Approach' },
	{ href: '/report/', label: 'Report' },
	{ href: '/team/', label: 'Team' },
]

export function SiteHeader() {
	const pathname = usePathname()
	const [open, setOpen] = React.useState(false)

	return (
		<header className='sticky top-0 z-40 border-b bg-background/85 backdrop-blur'>
			<div className='mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6'>
				<Link href='/' className='flex items-center gap-2 font-semibold tracking-tight'>
					<span className='flex size-7 items-center justify-center rounded-md bg-primary font-mono text-xs text-primary-foreground'>
						TA
					</span>
					<span>Traffic Analyzer</span>
				</Link>

				<nav aria-label='Main' className='ms-auto hidden items-center gap-1 md:flex'>
					{navItems.map((item) => (
						<Link
							key={item.href}
							href={item.href}
							className={cn(
								'rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground',
								pathname.startsWith(item.href) && 'bg-muted text-foreground',
							)}
						>
							{item.label}
						</Link>
					))}
				</nav>

				<div className='ms-auto flex items-center gap-1 md:ms-0'>
					<ThemeToggle />
					<Button
						variant='ghost'
						size='icon'
						className='md:hidden'
						aria-label={open ? 'Close menu' : 'Open menu'}
						aria-expanded={open}
						onClick={() => setOpen((value) => !value)}
					>
						{open ? <XIcon /> : <MenuIcon />}
					</Button>
				</div>
			</div>

			{open ? (
				<nav aria-label='Mobile' className='grid gap-1 border-t px-4 py-3 md:hidden'>
					{navItems.map((item) => (
						<Link
							key={item.href}
							href={item.href}
							onClick={() => setOpen(false)}
							className={cn(
								'rounded-md px-3 py-2 text-sm',
								pathname.startsWith(item.href) ? 'bg-muted font-medium' : 'text-muted-foreground',
							)}
						>
							{item.label}
						</Link>
					))}
				</nav>
			) : null}
		</header>
	)
}
