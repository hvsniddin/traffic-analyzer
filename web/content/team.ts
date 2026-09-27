// TODO(team): replace every "TODO" entry before the submission tag.
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
		role: 'TODO: role',
		contributions: ['TODO: what you built'],
		links: [
			{ label: 'GitHub', href: 'https://github.com/TODO' },
			{ label: 'LinkedIn', href: 'https://www.linkedin.com/in/TODO' },
		],
		projects: ['TODO: a previous project you are proud of'],
	},
	{
		name: 'TODO: name (hvsniddin)',
		role: 'TODO: role',
		contributions: ['TODO: what you built'],
		links: [{ label: 'GitHub', href: 'https://github.com/hvsniddin' }],
		projects: ['TODO: a previous project'],
	},
	{
		name: 'TODO: third member',
		role: 'TODO: role',
		contributions: ['TODO: what you built'],
		links: [],
		projects: [],
	},
]
