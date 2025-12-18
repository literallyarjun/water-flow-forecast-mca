# MetroFlow Design Guidelines

## Design Approach
**Design System Foundation**: Material Design for dashboards with data visualization focus
- Emphasizes clarity, readability, and structured information hierarchy
- Optimized for data-heavy applications with clear visual organization
- Teacher-friendly presentation suitable for classroom/presentation contexts

## Typography System
**Font Families**: 
- Primary: Inter or Roboto (clean, highly readable for data)
- Monospace: JetBrains Mono (for numerical values)

**Hierarchy**:
- Page Title: 2.5rem, semibold
- Section Headers: 1.75rem, medium
- Chart Titles: 1.25rem, medium
- Body/Labels: 1rem, regular
- Live Flow Number Display: 3rem, bold (hero number)
- Small Data Points: 0.875rem, regular

## Layout System
**Spacing Units**: Tailwind spacing of 4, 6, 8, 12, 16, 24
- Container padding: p-8
- Section spacing: mb-12
- Card padding: p-6
- Element gaps: gap-6 for grids, gap-4 for inline elements

**Grid Structure**:
- Dashboard: Single column on mobile, 12-column grid on desktop
- Climate indicators: 5-column grid (one per indicator)
- Charts: Full-width containers with max-w-7xl

## Page Layouts

### Home Page (Upload Interface)
**Structure**:
- Centered layout with max-w-2xl
- Large drag-and-drop upload zone (h-64)
- Dashed border for upload area
- Live Flow Level widget positioned prominently above upload (card style)
- Forecast range dropdown (w-full on mobile, w-64 on desktop)
- Large primary action button (w-full, h-14)

### Results Dashboard
**Component Hierarchy** (top to bottom):
1. **Header Bar**: Logo + Title + Download CSV button (sticky, h-16)
2. **Live Flow Widget**: Prominent card at top (h-32), flex layout with number left, sparkline right
3. **Main Consumption Chart**: Full-width, h-96, clear legend
4. **Climate Indicators Row**: 5 equal cards in grid, h-64 each
5. **Forecast Section**: Large chart (h-96) with period selector tabs
6. **Model Comparison**: Horizontal bar chart (h-80)
7. **Feature Importance**: Vertical bar chart (h-96)

## Component Library

### Cards
- Elevated appearance with subtle shadow
- Rounded corners (rounded-lg)
- Consistent padding (p-6)
- Header section with title + optional badge

### Live Flow Widget (Special)
**Layout**: 
- Large numerical display (left 60%, right-aligned)
- Mini sparkline chart (right 40%)
- Single-line label above number
- Subtle status indicator dot (normal/high/low)

### Charts
**Specifications**:
- White/light background for clarity
- Grid lines: subtle, horizontal only for line charts
- Axes: labeled clearly with simple units
- Legends: positioned top-right, horizontal layout
- No decimal places unless necessary

**Chart Types**:
- Line charts: 2px stroke width, smooth curves
- Bar charts: 24px bar width, 8px gaps
- Filled areas: 20% opacity for forecast shading

### Upload Zone
- Dashed border (border-2, border-dashed)
- Icon centered (cloud upload, 4rem size)
- "Drag & drop CSV or click to browse" text
- Hover state: subtle background shift
- Active drag: highlight border

### Buttons
**Primary Action** (Start Forecast, Download):
- Full-width on mobile, auto on desktop (min-w-48)
- Height h-12
- Medium font weight
- Rounded (rounded-md)

**Secondary Actions** (Period selectors):
- Tab-style buttons in horizontal group
- Equal width distribution
- Active state clearly distinguished

### Data Display Patterns
**Number Showcases** (Live Flow, Key Metrics):
- Large number with minimal decoration
- Label above or below (0.875rem)
- Optional unit in lighter weight if absolutely necessary
- Status indicator as subtle dot or badge

## Accessibility & Readability
- Minimum font size: 0.875rem (14px)
- Line height: 1.5 for body text, 1.2 for headings
- Chart labels: minimum 0.75rem, high contrast
- Focus states: 2px outline offset for all interactive elements
- ARIA labels for all charts and data visualizations

## Images
**No hero image required** - This is a data-focused dashboard application. Visual interest comes from:
- Charts and data visualizations
- Live flow sparkline animations
- Well-organized information architecture

## Visual Rhythm
**Vertical Flow**:
- Consistent section spacing (mb-12)
- Cards create visual grouping
- Charts alternate with text summaries
- White space prevents overwhelming density

**Horizontal Balance**:
- Climate indicators: equal-width columns
- Asymmetric layouts for number + chart combinations
- Centered upload interface for focus

## Critical Design Principles
1. **No Scientific Jargon**: All labels in plain language ("Water Level: 18" not "18 m³/s")
2. **Visual Comparison**: Use bar heights, line positions - not just numbers
3. **Teacher-Optimized**: Large enough to present on projector, high contrast
4. **Progressive Disclosure**: Most critical info (live flow, main trend) first
5. **Action Clarity**: One primary action per screen section