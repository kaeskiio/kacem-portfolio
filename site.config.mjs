// Everything on the home page lives here. Edit this file, run `npm run dev`, and refresh.
// Text fields accept inline Markdown (links, *italics*, `code`).

export default {
  name: "Kacem Ettahali",
  description:
    "Computer science student at Rice University working on reinforcement learning and AI safety.",
  email: "ke35@rice.edu",
  links: [
    { label: "GitHub", url: "https://github.com/kaeskiio" },
    { label: "LinkedIn", url: "https://www.linkedin.com/in/kacemettahali" },
  ],

  // The first thing visitors read. Keep it to one sentence.
  focus:
    "I'm working on how reinforcement learning changes a language model's behavior when its reward or oversight is imperfect.",

  intro:
    "I study computer science at Rice University, concentrating in artificial intelligence. I've worked on satellite algorithms at Lockheed Martin and built research software at MIT, Harvard, and UT Austin.",

  // Papers and research projects. Leave the list empty to hide the section.
  // Example entry:
  // {
  //   title: "Reward hacking dynamics during GRPO post-training",
  //   url: "https://github.com/kaeskiio/rl-safety-lab",
  //   venue: "In progress",
  //   summary: "Tracks how proxy and ground-truth reward diverge across training checkpoints.",
  // },
  research: [],

  experience: [
    { when: "2025–now", org: "MIT & Harvard", role: "AI/ML and full-stack software engineer" },
    { when: "Summer 2025", org: "Lockheed Martin", role: "Software engineer" },
    { when: "2024–25", org: "University of Texas at Austin", role: "Research software engineer" },
    { when: "Summer 2024", org: "Rice University PATHS-UP", role: "Instructor" },
  ],

  projects: [
    {
      title: "Hand sign recognition",
      summary:
        "Webcam classifier for 12 gestures at 98% accuracy, later adapted to translate ASL signs into captions.",
    },
    {
      title: "Chemistry Memory",
      url: "https://chemistryquiz.net/",
      summary: "Practice tools that help chemistry students memorize what they need to.",
    },
    {
      title: "Pixel Reveal",
      url: "https://github.com/kacemettahali/computerVisionGame",
      summary: "A guessing game where an image appears pixel by pixel, built to explore computer vision.",
    },
    {
      title: "Shadow Creek Tennis Camp",
      url: "https://www.sctenniscamp.org/",
      summary: "Website for a summer tennis camp.",
    },
  ],

  // Shown as a short closing paragraph on the home page. Set to "" to hide.
  outside:
    "Outside of research I'm Executive VP of Directors at [First Racquet](https://firstracquet.org), where we run fundraisers and charity tournaments that send tennis equipment to young athletes in Luanda, Angola.",
};
