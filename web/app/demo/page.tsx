import type { Metadata } from 'next'

import { DemoConsole } from '@/components/demo/demo-console'
import { PageHeader } from '@/components/shared/page-header'

export const metadata: Metadata = { title: 'Live demo' }

export default function DemoPage() {
	return (
		<>
			<PageHeader kicker='Live demo' title='Upload a clip, get the traffic events back'>
				The same <code className='font-mono text-sm'>analyze_video</code> pipeline the offline submission uses runs on
				the server. When it finishes you get every event as a time segment with a class. Click any segment to jump the
				video to it.
			</PageHeader>
			<DemoConsole />
		</>
	)
}
