import { ExternalLinkIcon } from 'lucide-react'
import type { Metadata } from 'next'

import { PageHeader } from '@/components/shared/page-header'
import { Card, CardContent } from '@/components/ui/card'
import { members, teamName } from '@/content/team'

export const metadata: Metadata = { title: 'Team' }

function initials(name: string) {
	return name
		.replace(/\(.*\)/, '')
		.split(/\s+/)
		.filter(Boolean)
		.slice(0, 2)
		.map((part) => part[0]?.toUpperCase())
		.join('')
}

export default function TeamPage() {
	return (
		<>
			<PageHeader kicker='Team' title={teamName}>
				Three people, one repository. Who did what is listed here and in the README.
			</PageHeader>
			<div className='grid gap-4 md:grid-cols-3'>
				{members.map((member) => (
					<Card key={member.name}>
						<CardContent className='grid gap-4'>
							<div className='flex items-center gap-3'>
								<span className='flex size-11 shrink-0 items-center justify-center rounded-full bg-muted font-semibold'>
									{initials(member.name)}
								</span>
								<div className='min-w-0'>
									<p className='font-semibold'>{member.name}</p>
									<p className='text-sm text-muted-foreground'>{member.role}</p>
								</div>
							</div>
							<div className='grid gap-1.5'>
								<p className='text-xs font-medium tracking-wide text-muted-foreground uppercase'>Contributed</p>
								<ul className='grid list-disc gap-1 ps-5 text-sm'>
									{member.contributions.map((item) => (
										<li key={item}>{item}</li>
									))}
								</ul>
							</div>
							{member.projects.length ? (
								<div className='grid gap-1.5'>
									<p className='text-xs font-medium tracking-wide text-muted-foreground uppercase'>Previous work</p>
									<ul className='grid list-disc gap-1 ps-5 text-sm'>
										{member.projects.map((item) => (
											<li key={item}>{item}</li>
										))}
									</ul>
								</div>
							) : null}
							{member.links.length ? (
								<div className='flex flex-wrap gap-3 text-sm'>
									{member.links.map((link) => (
										<a
											key={link.href}
											href={link.href}
											className='inline-flex items-center gap-1 text-primary hover:underline'
										>
											{link.label} <ExternalLinkIcon className='size-3' />
										</a>
									))}
								</div>
							) : null}
						</CardContent>
					</Card>
				))}
			</div>
		</>
	)
}
