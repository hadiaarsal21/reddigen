// Inline SVG icon set. Matches the visual style of the production
// ReddiGen icons: stroke-based line icons at strokeWidth 1.8, rounded caps.

import type { SVGProps } from 'react';

type IconName =
  | 'home'
  | 'search'
  | 'list'
  | 'compass'
  | 'layers'
  | 'brain'
  | 'gem'
  | 'external'
  | 'check'
  | 'x'
  | 'menu'
  | 'trash'
  | 'refresh'
  | 'sparkles'
  | 'chevron'
  | 'calendar'
  | 'target'
  | 'message'
  | 'sliders'
  | 'arrow-right'
  | 'alert'
  | 'copy';

interface Props extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 16, ...rest }: Props) {
  const common = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    ...rest,
  };
  switch (name) {
    case 'home':
      return (
        <svg {...common}>
          <path d="M3 12l9-9 9 9M5 10v10a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-4h2v4a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1V10" />
        </svg>
      );
    case 'search':
      return (
        <svg {...common}>
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-4.35-4.35" />
        </svg>
      );
    case 'list':
      return (
        <svg {...common}>
          <path d="M8 6h13M8 12h13M8 18h13" />
          <circle cx="4" cy="6" r="1" fill="currentColor" />
          <circle cx="4" cy="12" r="1" fill="currentColor" />
          <circle cx="4" cy="18" r="1" fill="currentColor" />
        </svg>
      );
    case 'compass':
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="m15.5 8.5-2 5-5 2 2-5 5-2z" />
        </svg>
      );
    case 'layers':
      return (
        <svg {...common}>
          <path d="m12 2 9 5-9 5-9-5 9-5z" />
          <path d="m3 12 9 5 9-5" />
          <path d="m3 17 9 5 9-5" />
        </svg>
      );
    case 'brain':
      return (
        <svg {...common}>
          <path d="M9 3a3 3 0 0 1 3 3v12a3 3 0 0 1-6 0V9a3 3 0 0 1 3-3z" />
          <path d="M15 3a3 3 0 0 0-3 3v12a3 3 0 0 0 6 0V9a3 3 0 0 0-3-3z" />
          <path d="M12 8v0M12 14v0" />
        </svg>
      );
    case 'gem':
      return (
        <svg {...common}>
          <path d="M6 3h12l4 6-10 13L2 9z" />
          <path d="M11 3 8 9l4 13 4-13-3-6M2 9h20" />
        </svg>
      );
    case 'external':
      return (
        <svg {...common}>
          <path d="M14 3h7v7" />
          <path d="M10 14 21 3" />
          <path d="M21 14v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h6" />
        </svg>
      );
    case 'check':
      return (
        <svg {...common} strokeWidth={2.4}>
          <path d="m5 12 4.5 4.5L19 7" />
        </svg>
      );
    case 'x':
      return (
        <svg {...common}>
          <path d="M6 6l12 12M18 6 6 18" />
        </svg>
      );
    case 'menu':
      return (
        <svg {...common}>
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      );
    case 'trash':
      return (
        <svg {...common}>
          <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M6 6l1 14a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-14" />
        </svg>
      );
    case 'refresh':
      return (
        <svg {...common}>
          <path d="M23 4v6h-6M1 20v-6h6" />
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
      );
    case 'sparkles':
      return (
        <svg {...common}>
          <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />
        </svg>
      );
    case 'chevron':
      return (
        <svg {...common}>
          <path d="m6 9 6 6 6-6" />
        </svg>
      );
    case 'calendar':
      return (
        <svg {...common}>
          <rect x="3" y="5" width="18" height="16" rx="2" />
          <path d="M3 10h18M8 3v4M16 3v4" />
        </svg>
      );
    case 'target':
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <circle cx="12" cy="12" r="4" />
          <circle cx="12" cy="12" r="1" fill="currentColor" />
        </svg>
      );
    case 'message':
      return (
        <svg {...common}>
          <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9 9 0 0 1-3.9-.9L3 21l1.9-5.1A8.4 8.4 0 0 1 4 11.5a8.5 8.5 0 0 1 8.5-8.5A8.4 8.4 0 0 1 21 11.5z" />
        </svg>
      );
    case 'sliders':
      return (
        <svg {...common}>
          <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
          <path d="M1 14h6M9 8h6M17 16h6" />
        </svg>
      );
    case 'arrow-right':
      return (
        <svg {...common}>
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      );
    case 'alert':
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v6M12 16v0" />
        </svg>
      );
    case 'copy':
      return (
        <svg {...common}>
          <rect x="9" y="9" width="12" height="12" rx="2" />
          <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
        </svg>
      );
    default:
      return null;
  }
}
