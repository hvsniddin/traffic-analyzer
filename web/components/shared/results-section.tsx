import {
	ablations,
	falsePositives,
	finalScores,
	labelOutcomes,
	minLengthSweep,
	outcomesByClass,
	perClass,
} from '@/content/results'

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
	return (
		<div className='not-prose my-4 overflow-x-auto rounded-xl border'>
			<table className='w-full text-sm'>
				<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
					<tr>
						{head.map((cell, index) => (
							<th key={cell} className={index === 0 ? 'px-3 py-2 font-medium' : 'px-3 py-2 text-right font-medium'}>
								{cell}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{rows.map((row) => (
						<tr key={row.join('|')} className='border-t'>
							{row.map((cell, index) => (
								<td
									key={index}
									className={
										index === 0 ? 'px-3 py-2' : 'px-3 py-2 text-right font-mono tabular-nums'
									}
								>
									{typeof cell === 'number' && !Number.isInteger(cell) ? cell.toFixed(3) : cell}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
		</div>
	)
}

export function ResultsSection() {
	const total = labelOutcomes.matched + labelOutcomes.boundary + labelOutcomes.missed
	return (
		<>
			<h2 id='results'>Results on our dev labels</h2>
			<p>
				Scored with the unchanged <code>evaluate.py</code> against our own labels of the four sample clips (
				<code>annotations/ground_truth.json</code>, {total} events). <strong>Score A = {finalScores.scoreA}</strong>{' '}
				for <code>predictions_samples.json</code>. Micro F1 is {finalScores.micro['0.3']} / {finalScores.micro['0.5']}{' '}
				/ {finalScores.micro['0.7']} at IoU 0.3 / 0.5 / 0.7; ignoring labels it is {finalScores.classAgnostic['0.3']}{' '}
				/ {finalScores.classAgnostic['0.5']} / {finalScores.classAgnostic['0.7']}. Part B cannot be scored here: the
				samples contain no accidents.
			</p>
			<Table
				head={['Class', 'F1 @0.3', 'F1 @0.5', 'F1 @0.7', 'TP', 'FP', 'FN']}
				rows={perClass.map((row) => [...row])}
			/>

			<h3>Ablations</h3>
			<p>Each step adds to the one above it; Score A on the same four clips.</p>
			<Table
				head={['Change', 'Score A', 'Note']}
				rows={ablations.map(({ step, scoreA, note }) => [step, scoreA, note ?? ''])}
			/>
			<p>
				Frame rate, on C3905 with the earlier rules: 0.5 FPS scored 0.000 and 5 FPS 0.022. We now sample 2 FPS on
				every device, so CPU and GPU runs give the same events. Minimum segment length:
			</p>
			<Table
				head={['Minimum (s)', 'Score A', 'Mean F1 @0.7']}
				rows={minLengthSweep.map((row) => [...row])}
			/>
			<p>
				We keep 1 s: longer minimums gain little and would delete genuine 1 s events such as red_light and short
				failure_to_yield crossings.
			</p>

			<h3>Error analysis</h3>
			<p>
				Of the {total} labelled events, {labelOutcomes.matched} are matched at IoU 0.5, {labelOutcomes.boundary} are
				found but with boundaries too far off to match, and {labelOutcomes.missed} have no same-class prediction at
				all. Jaywalking is mostly a boundary problem; failure_to_yield is mostly missed.
			</p>
			<Table
				head={['Class', 'Matched', 'Boundaries off', 'Missed']}
				rows={outcomesByClass.map((row) => [...row])}
			/>
			<p>
				Where the unmatched predictions fall. Most are the right class at the wrong time span; the main confusion is
				between pedestrian classes, since failure_to_yield and near_miss fire while someone is labelled as jaywalking.
			</p>
			<Table head={['Predicted class', 'What it overlaps', 'Count']} rows={falsePositives.map((row) => [...row])} />
			<p>
				<strong>Why the model reports 44 events against 80 labels.</strong> Label merging is not the cause: merging
				overlapping same-class labels only took our 84 raw labels to 80. Most of the gap is failure_to_yield, where the
				model finds 11 events against 27 labelled and covers 37 s of the 221 s we labelled. Jaywalking is seen about as
				much (731 s predicted, 685 s labelled) but in fewer, longer segments (median 18 s against 8.5 s), because the
				8 s merge gap joins nearby pedestrians. Four labelled classes are never predicted, and the 1 s minimum drops 16
				short segments.
			</p>
			<p>
				Our labels start and end on whole seconds, so each boundary can be up to 0.5 s off while predictions are
				frame-exact. For short events such as a 2 s failure_to_yield, rounding alone can push a correct prediction
				below IoU 0.7. Per-clip mismatches are listed under <em>Failure cases</em> on each sample page.
			</p>
		</>
	)
}
