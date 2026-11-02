# What a fresh clone can reproduce

The repository includes the generic narrator, model/tool interfaces, input backends, Chrono
Trigger's pixel-reader profile and small glyph/cursor templates, plus synthetic arena/town
fixtures. With a key and a supported desktop, a judge can try generic descriptions, questions
and input tools. Without a key, the offline smoke check exercises reader loading and planning.

**The measured map navigation and seller approach cannot currently be reproduced from
this repository alone.** The README's live navigation/shop trials used additional local data that is
not included. Installing or owning Chrono Trigger does not populate these files automatically.

## Preparation versus runtime

Preparation tools run a separately supplied game in an emulator. They read emulator RAM for
player/camera coordinates, collision, locations and exit transitions, and stitch captured
images into maps and room pictures. Muse labels destinations/people/exits and drafts atmosphere
text; a human checks it. Benchmark tools also use emulator memory to judge task completion.
This is memory-assisted preparation and evaluation, not learning all navigation from pixels.

The Steam runtime reads screen pixels. It matches the view against the prepared images,
tracks the player and plans/checks movement on surveyed walkable cells. The current Steam
navigation path does not read Steam process memory. The packaged pixel reader likewise reads
text, highlighted choices and HP from screenshots, using its learned glyph/template library.

## Missing from the demonstrated configuration

- Atlas/reference map images, labelled destinations and walkability/transition data.
- Room JSON files and their stitched reference pictures, room names and connections.
- Player sprite templates/anchors and view-scale configuration used for localization.
- Game packs, authored/derived notes and walkthrough/quest data used in that session.
- The particular saves, private recordings and bench ground-truth files.

The runtime expects these under the user's player-two configuration directories. Survey
scripts are present, but running them requires an emulator, a separately obtained lawful game
copy, setup and curation. They are not a two-minute end-user installer for the recorded demo.

## What could be distributed

The project can ship its own code, synthetic fixtures and original instructions under the
project license. A candidate data-only package could contain minimal walkability grids,
coordinates, connectivity and human-written labels/instructions, with provenance and a separate
license. Those are lower-artwork candidates, not a determination of redistribution rights:
review the source, publisher terms and any copied names/text before release. No package has
been approved or distributed.

**Removing pictures from JSON does not make the demo work.** The current localizer still needs
reference pictures and player templates. A possible owner-supplied workflow would generate those
locally from the player's game and pair them with reviewed data-only grids. That needs additional
engineering and clean-machine validation; it is not implemented or claimed here.

Do not bundle fan map scans, stitched game screenshots, sprites, fonts, walkthrough text, ROMs
or saves merely because the developer owns a copy. Fan-map authors and walkthrough authors
may have rights separate from the publisher. The already-shipped reader glyph/cursor templates
also retain their underlying game rights; MIT covers the project code, not that artwork.
Footage-sharing policies are not asset-redistribution licenses.
