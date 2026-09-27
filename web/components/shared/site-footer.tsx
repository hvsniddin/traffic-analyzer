import Link from 'next/link'

import { config } from '@/lib/config'

export function SiteFooter() {
	return (
		<footer className='mt-20 border-t'>
			<div className='mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6'>
				<p>WIUT Hackathon 2026 · Computer Vision elimination task</p>
				<div className='flex flex-wrap gap-4'>
					<a href={config.repoUrl} className='hover:text-foreground'>
						Repository
					</a>
					<Link href='/report/#links' className='hover:text-foreground'>
						Weights and predictions
					</Link>
					<Link href='/team/' className='hover:text-foreground'>
						Team
					</Link>
				</div>
			</div>
		</footer>
	)
}
