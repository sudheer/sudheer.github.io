import { defineCollection, z } from 'astro:content';

const baseSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  pubDate: z.coerce.date(),
  updatedDate: z.coerce.date().optional(),
  tags: z.array(z.string()).default([]),
  draft: z.boolean().default(false)
});

const posts = defineCollection({
  schema: baseSchema.extend({
    category: z.enum(['Writing', 'Notes', 'TIL', 'Reading', 'Projects']).default('Writing')
  })
});

const micro = defineCollection({
  schema: baseSchema.extend({
    category: z.literal('Micro').default('Micro')
  })
});

export const collections = { posts, micro };
