import type { Metadata } from 'next'
import Link from 'next/link'

import { PageHeader } from '@/components/shared/page-header'
import { config } from '@/lib/config'

export const metadata: Metadata = { title: 'Technical report' }

export default function ReportPage() {
	const blob = `${config.repoUrl}/blob/master`

	return (
		<>
			<PageHeader kicker='Technical report' title='What we built, what worked, what did not'>
				One page, written for someone who wants to rebuild the system or decide what to fix next.
			</PageHeader>

			<article className='prose-page'>
				<h2>What we built</h2>
				<p>
					A detector-tracker-rules pipeline behind the official <code>solution.detect_events</code> interface. A
					fine-tuned YOLO finds road users and signal heads, ByteTrack links them into tracks, and a scene map of the
					junction is aligned to each clip with a SIFT homography. Seven per-class rule modules turn tracks and zones
					into time segments. The same entry point backs the live demo. See the <Link href='/approach/'>approach</Link>{' '}
					for the diagram and every rule.
				</p>

				<h2>What worked</h2>
				<ul>
					<li>
						<strong>Signal heads as detector classes.</strong> Red and green heads come out of the same detector pass,
						so red_light and stop_line need no extra model. The phase strip on the <Link href='/eda/'>EDA page</Link>{' '}
						shows when each head is visible.
					</li>
					<li>
						<strong>Measuring the camera shift before trusting the map.</strong> The &ldquo;fixed&rdquo; camera moved
						by up to 38 px (at 720p) between clips. The alignment step brings the residual under 0.5 px on every sample,
						and it falls back to the original map with a logged reason when a match is unreliable.
					</li>
					<li>
						<strong>One shared scene map.</strong> Lanes, crossings, island and stop lines drawn once cover all 14
						classes&apos; spatial needs and carry over to the hidden set from the same camera.
					</li>
				</ul>

				<h2>What did not work, or not yet</h2>
				<ul>
					<li>
						<strong>Seven classes have no detector:</strong> accident, near_miss, wrong_way, illegal_u_turn,
						stopped_vehicle, road_obstacle and fire_smoke. Part B (accident anticipation) is not implemented, so its
						score is 0.
					</li>
					<li>
						<strong>Our dev set is small.</strong> We manually labelled all four provided clips and merged overlapping
						segments of the same class. It contains 22, 18, 29 and 11 event intervals respectively. Results on these
						clips may not generalise to unseen traffic or weather.
					</li>
					<li>
						<strong>Label definitions are ambiguous at the edges.</strong> Our first pass had long failure_to_yield
						and jaywalking spans that merged many people. Under the task&apos;s one-segment-per-overlap convention these
						become single long events, which is correct but hard to match at IoU 0.7.
					</li>
					<li>
						<strong>4K decoding dominates runtime on CPU.</strong> Building this site&apos;s 540p previews from the
						140 Mbit/s originals took several times real time on a 4-core laptop, before any detector ran. The GPU judging machine has the budget, but the CPU demo
						must limit clip length.
					</li>
					<li>
						<strong>Part of C3896 does not decode cleanly.</strong> ffmpeg reports corrupt H.264 packets in the
						original file. Decoding recovers and the clip plays to the end, but some frames can come out damaged, so a
						reader should expect failed reads and rules should not treat a missing frame as a stopped vehicle.
					</li>
				</ul>

				<h2>What we would do next</h2>
				<ol>
					<li>Have a second reviewer audit the four labelled clips, then tune thresholds against F1 at IoU 0.7.</li>
					<li>
						wrong_way and stopped_vehicle next: both are direct rules on existing tracks plus the lane directions we
						already measured with optical flow.
					</li>
					<li>
						Part B with time-to-collision between track pairs, calibrated so 0.5 means &ldquo;probably within 5
						s&rdquo;. This also gives near_miss candidates.
					</li>
					<li>Decode on the GPU (NVDEC) and batch the detector to widen the time margin.</li>
				</ol>

				<h2 id='links'>Links</h2>
				<ul>
					<li>
						Repository: <a href={config.repoUrl}>{config.repoUrl}</a>
					</li>
					<li>
						Sample predictions: <a href={`${blob}/predictions_samples.json`}>predictions_samples.json</a>
					</li>
					<li>
						Weights: <a href={`${config.repoUrl}/tree/master/weights`}>weights/</a>
					</li>
					<li>
						Demo API contract: <a href={`${blob}/README.md`}>README</a>
					</li>
				</ul>
			</article>
		</>
	)
}
