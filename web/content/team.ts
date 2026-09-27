export type Member = {
	name: string
	role: string
	contributions: string[]
	links: { label: string; href: string }[]
	projects: string[]
}

export const teamName = 'Lorem Ipsum'

export const members: Member[] = [
	{
		name: "Ja'farbek Yusupov",
		role: 'Scene mapping, event logic, and integration',
		contributions: ['Manual event labels and fixed-camera scene layout', 'Event rules, backend integration, and evaluation'],
		links: [
			{ label: 'Website', href: 'https://jafarbekyusupov.com' },
			{ label: 'GitHub', href: 'https://github.com/jafarbekyusupov' },
			{ label: 'LinkedIn', href: 'https://www.linkedin.com/in/jafarbekyusupov' },
		],
		projects: [],
	},
	{
		name: 'Husniddin Ravshanov',
		role: 'Model training and event detection',
		contributions: ['Detector testing and fine-tuning', 'Event detection rules and solution implementation'],
		links: [{ label: 'GitHub', href: 'https://github.com/hvsniddin' }],
		projects: [],
	},
	{
		name: 'Azizbek Rakhmatulloyev',
		role: 'Frontend and annotation',
		contributions: ['Website frontend', 'Bounding-box annotations for detector fine-tuning'],
		links: [{ label: 'GitHub', href: 'https://github.com/aRakhmatullaev' }],
		projects: [],
	},
]
