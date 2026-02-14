# Video Tutorial Scripts

Production-ready scripts for creating Transistor Database video tutorials.

## Video Series Overview

**Target audience**: Power electronics engineers, researchers, students

**Recommended format**: Screen recording with voiceover, 1080p, 30fps

**Suggested platform**: YouTube, with chapters/timestamps

---

## Video 1: Introduction & Quick Tour (5 minutes)

### Script

**[00:00 - Opening]**

> "Welcome to the Transistor Database - an open-source tool for managing power semiconductor data and analyzing converter designs. I'm [Your Name], and in this video, I'll give you a quick tour of the interface and show you how to get started."

**[00:15 - Show homepage]**

> "The Transistor Database comes in two flavors: a desktop PyQt5 interface and a modern web interface. Today we'll use the web interface, which runs on localhost:5173 with a FastAPI backend on port 8002."

**[00:30 - Show main navigation]**

> "At the top, we have six main tabs. Let's go through each one quickly."

**[00:45 - Search Database tab]**

> "First, Search Database. This is where you'll spend most of your time. You can filter transistors by type, voltage rating, current rating, housing type, and more. Let's filter for SiC MOSFETs rated above 1000 volts."

**[01:15 - Apply filters, show results]**

> "Great! We found several devices. Notice the table shows key specifications: voltage rating, current rating, on-resistance, and thermal resistance. Click any transistor name to see complete details."

**[01:45 - Create Transistor tab]**

> "Second tab: Create Transistor. This is where you add new devices to your database. You'll need the datasheet handy. Fill in metadata like name and manufacturer, then add electrical ratings, thermal properties, and characteristic curves."

**[02:15 - Exporting Tools tab]**

> "Third: Exporting Tools. This is a killer feature. Select a transistor, choose your simulation tool - PLECS, MATLAB, Simulink, or GeckoCIRCUITS - and export. The tool generates properly formatted files ready to import into your simulator."

**[02:45 - Comparison Tools tab]**

> "Fourth: Comparison Tools. Select two or three transistors and compare them side-by-side. You get charts for on-resistance, switching losses, capacitance curves - everything you need to make an informed choice."

**[03:15 - Topology Calculator tab]**

> "Fifth: Topology Calculator. This is where it gets really cool. Select a converter topology - Buck, Boost, or Buck-Boost - choose your transistor, input your specifications, and the calculator predicts efficiency, losses, and junction temperature."

**[03:45 - Settings tab]**

> "Finally, Settings. Configure your database path, export preferences, and display options."

**[04:00 - Closing]**

> "That's the quick tour! In the next videos, we'll dive deeper into each feature. The database comes with sample transistors, and you can add your own from datasheets. It's completely free and open source on GitHub."

**[04:20 - Call to action]**

> "If you found this helpful, like and subscribe! Check the description for installation instructions and documentation links. See you in the next video!"

**[04:30 - End card]**

---

## Video 2: Searching & Filtering (8 minutes)

### Script

**[00:00 - Opening]**

> "Welcome back! Today we're diving deep into the Search Database feature. By the end of this video, you'll be able to quickly find the perfect transistor for your design."

**[00:15 - Show Search tab]**

> "Let's start with a real-world scenario: I'm designing a 400-volt Buck converter with 50 amps output current. I need a SiC MOSFET with good efficiency."

**[00:30 - Type filter]**

> "First, enable the Type filter and select SiC-MOSFET. SiC devices have lower switching losses than silicon, which is perfect for high-frequency converters."

**[01:00 - Voltage filter]**

> "Next, voltage rating. My DC bus is 400 volts, but I want 50% margin for safety. So I need at least 600 volts. Enable the Voltage Rating filter, set minimum to 600, and leave maximum blank to see all options."

**[01:30 - Current filter]**

> "For current: my output is 50 amps, but peak current during transients could be higher. Let's filter for at least 60 amps continuous current."

**[02:00 - Show results]**

> "Perfect! We have several candidates. Notice the table is sortable - click any column header to sort. Let's sort by on-resistance to find the most efficient option."

**[02:30 - Click top result]**

> "Here's our winner: the CREE C3M0060065J. 650 volts, 60 amps continuous, only 6 milliohms on-resistance. Click the name to see full details."

**[03:00 - Show details view]**

> "Look at all this data! Electrical ratings, thermal properties, switching characteristics, capacitance curves. All extracted from the manufacturer's datasheet."

**[03:30 - Additional filters]**

> "Back to search. We have more filters available: housing type if you need a specific footprint, manufacturer if you have a preferred supplier, and name search if you know exactly what you're looking for."

**[04:15 - Housing filter demo]**

> "Let's say I need a TO-247 package for easy heatsink mounting. Enable Housing Type, select TO-247-3, and re-search. Now we're seeing only TO-247 devices."

**[04:45 - Reset filters]**

> "The Reset Filters button clears everything and shows the full database again. Very handy when you want to start over."

**[05:00 - Export filtered results]**

> "Here's a pro tip: after filtering, you can export the entire result set. Click Export Results, choose JSON, and you have a backup of your filtered selection."

**[05:30 - Name contains filter]**

> "One more trick: the Name Contains filter is great when you remember part of a device name. Type 'IPT' and it shows all Infineon IPT-series transistors."

**[06:00 - Practical tips]**

> "Some practical tips for effective searching: First, start broad and narrow down. Don't enable all filters at once or you might get zero results. Second, use voltage and current filters first - these eliminate the most options. Third, sort results by the parameter that matters most for your application."

**[06:45 - Common mistakes]**

> "Common mistake: setting voltage exactly to your bus voltage. Remember, you need headroom! I recommend at least 30-50% margin. For a 400V bus, search for 600V devices minimum."

**[07:15 - Closing]**

> "That's everything you need to master the search function! In the next video, we'll compare multiple transistors side-by-side to make the final selection. Thanks for watching!"

**[07:30 - End card]**

---

## Video 3: Comparing Transistors (10 minutes)

### Script

**[00:00 - Opening]**

> "You've filtered down to three candidate transistors, but which one should you actually use? That's where the Comparison Tools come in. Let's compare devices and make an informed decision."

**[00:15 - Navigate to Comparison tab]**

> "Switch to the Comparison Tools tab. You'll see three dropdown menus for selecting transistors to compare."

**[00:30 - Select first transistor]**

> "From our previous search, we shortlisted three SiC MOSFETs. Let's compare them: CREE C3M0060065J, Infineon IPT60R028G7, and Rohm SCT3040KL."

**[01:00 - Show comparison table]**

> "Boom! Instant side-by-side comparison. The table shows every important parameter. Notice the color coding: green for best, yellow for middle, red for worst in each category."

**[01:30 - Electrical ratings]**

> "Let's analyze the electrical ratings. All three are 650V class, so voltage is not a differentiator. But look at current ratings: CREE is 60A continuous, Infineon 45A, Rohm 50A. CREE wins on current capability."

**[02:00 - On-resistance]**

> "Now on-resistance, which determines conduction losses. CREE: 6 milliohms. Infineon: 2.8 milliohms - that's fantastic! Rohm: 4 milliohms. For conduction losses, Infineon is the winner."

**[02:30 - Switching characteristics]**

> "But we also need to consider switching losses. Scroll down to the switching characteristics chart. This shows turn-on and turn-off energy at different currents."

**[03:00 - Analyze switching loss chart]**

> "At 50 amps: CREE has 0.8 millijoules turn-on, Infineon 1.2 millijoules, Rohm 0.9 millijoules. CREE wins on switching losses. This is the classic SiC tradeoff: Infineon has lower conduction loss but higher switching loss."

**[03:45 - Thermal comparison]**

> "Thermal resistance is critical for reliability. CREE: 0.24 K/W junction-to-case. Infineon: 0.30 K/W. Rohm: 0.26 K/W. CREE has the best thermal performance."

**[04:15 - Capacitance comparison]**

> "Output capacitance affects switching speed and losses. Check the capacitance chart. At 400V: CREE has 350 picofarads, Infineon 280pF, Rohm 320pF. Lower is better for fast switching, so Infineon wins here."

**[04:45 - Cost considerations note]**

> "Now, the database doesn't include pricing - that changes too frequently - but typically, devices with better specs cost more. Infineon's low on-resistance probably comes with a price premium."

**[05:15 - Application-specific decision]**

> "So which transistor should you choose? It depends on your application! If you're running high frequency - say, 100 kHz or above - switching losses dominate, so choose CREE. If you're at lower frequency with high average current, choose Infineon for the lower conduction loss."

**[06:00 - Export comparison]**

> "You can export this comparison as a PDF report or Excel spreadsheet. Great for design documentation or sharing with your team."

**[06:30 - Pop-out charts]**

> "Each chart has a pop-out button. Click it to see the chart in full screen. Perfect for presentations or closer analysis."

**[07:00 - Gate resistance slider]**

> "Here's a cool feature: the gate resistance slider. Gate resistance affects switching speed and losses. Drag the slider to see how different gate resistances change the comparison. Lower gate resistance means faster switching but more ringing."

**[07:45 - Clear and reset]**

> "The Clear button removes all selections so you can start a fresh comparison. Handy when you want to compare a different set of devices."

**[08:00 - Practical tips]**

> "Pro tips: First, compare devices in the same voltage class. Don't compare a 650V SiC with a 1200V IGBT - they're for different applications. Second, pay attention to test conditions. Switching losses are measured at specific voltage, current, and gate resistance. Make sure they're comparable."

**[08:45 - Closing]**

> "That's how you compare transistors like a pro! Next video, we'll take our chosen device and export it to PLECS for thermal simulation. See you then!"

**[09:00 - End card]**

---

## Video 4: Adding Your Own Transistor (15 minutes)

### Script

**[00:00 - Opening]**

> "You've found the perfect transistor for your design, but it's not in the database yet. No problem! In this video, I'll show you how to add a new transistor from the datasheet. We'll go through the entire process step-by-step."

**[00:20 - Show datasheet]**

> "I'm adding the Wolfspeed C3M0025065K - a 650V, 90A SiC MOSFET. I have the datasheet open in another window. You can download datasheets from manufacturer websites. Search for the part number plus 'datasheet'."

**[00:45 - Navigate to Create Transistor]**

> "Switch to the Create Transistor tab. The form is organized into sections: Metadata, Electrical Ratings, Thermal Properties, Switch Characteristics, Diode Characteristics, and Capacitances."

**[01:15 - Fill metadata]**

> "Let's start with metadata. Name: use the format Manufacturer_PartNumber, so 'Wolfspeed_C3M0025065K'. Type: this is a SiC MOSFET, so select that from the dropdown. Manufacturer: Wolfspeed. Housing: look at the datasheet... it's TO-247-3."

**[02:00 - Datasheet link]**

> "Datasheet hyperlink: copy the URL from the manufacturer's website and paste it here. This way you can always get back to the original datasheet."

**[02:20 - Electrical ratings from datasheet]**

> "Now electrical ratings. Open the datasheet to the 'Maximum Ratings' table, usually on page 2. V_abs_max: the datasheet says 650 volts. I_abs_max: 285 amps pulsed. I_cont: 90 amps at 25 degrees Celsius. T_j_max: 175 degrees Celsius. All SiC devices can handle high temperatures."

**[03:15 - Thermal properties]**

> "Thermal properties: find the 'Thermal Resistance' section. R_th_junction_case: 0.25 K/W. This is critical for thermal calculations. R_th_case_heatsink: usually 0 if you're using thermal paste. Leave blank if not specified."

**[03:45 - On-resistance note]**

> "The form shows R_ds_on fields for different temperatures. From the datasheet: 2.5 milliohms at 25°C with 15V gate voltage. At 150°C: about 3.8 milliohms. Temperature dependence is important for accurate loss calculations."

**[04:30 - Switch characteristics intro]**

> "Now the fun part: characteristic curves! These curves are what make the database so powerful. Let's add the channel characteristics first."

**[05:00 - Add channel characteristic]**

> "Click 'Add Channel Characteristic'. We need to digitize the I-V curve from the datasheet. Find the 'Output Characteristics' graph - shows drain current versus drain-source voltage at different gate voltages."

**[05:30 - Digitizing the curve]**

> "Here's how to digitize: Look at the curve for V_gs = 15V at T_j = 25°C. Pick 5-7 points along the curve. At V_ds = 0V, I_d = 0A. At 0.5V, about 10A. At 1V, about 20A. At 1.5V, about 28A. At 2V, it's saturated at 30A. Enter these values in the form."

**[06:30 - Multiple curves]**

> "Repeat for the high-temperature curve at 150°C. More curves mean more accurate simulations, but even two curves (25°C and 150°C) give good results."

**[07:00 - Switching losses]**

> "Next: switching losses. Find the 'Switching Energy vs Current' graphs. There are two: E_on for turn-on energy and E_off for turn-off energy."

**[07:30 - E_on curve]**

> "Click 'Add E_on Data'. Test conditions matter! The datasheet shows measurements at 400V bus voltage, 15V gate voltage, 10 ohm gate resistance, 25°C. Enter these first. Then digitize the E_on curve: at 10A, 0.2 millijoules. At 30A, 0.8 millijoules. At 50A, 1.5 millijoules. And so on."

**[08:30 - E_off curve]**

> "Repeat for E_off. Usually E_off is slightly lower than E_on for SiC MOSFETs."

**[09:00 - Diode characteristics]**

> "Don't skip the diode characteristics! SiC MOSFETs have body diodes. Find the 'Diode Forward Characteristics' graph. Same process: digitize voltage versus current at different temperatures."

**[09:45 - Reverse recovery]**

> "And reverse recovery energy E_rr. This is important for hard-switching converters. Find the E_rr graph and digitize it."

**[10:30 - Capacitances]**

> "Almost done! Add capacitance curves. Find the 'Capacitance vs Voltage' graph. There are three capacitances: C_oss (output), C_iss (input), and C_rss (reverse transfer). Digitize each one."

**[11:15 - Gate charge]**

> "Finally, gate charge curves if available. Find the 'Gate Charge vs Gate Voltage' graph. These curves show how the gate capacitance charges during switching."

**[11:45 - Review before saving]**

> "Before saving, scroll through and double-check everything. Common mistakes: wrong decimal places (microohms instead of milliohms), mixing up temperatures, missing units."

**[12:15 - Save transistor]**

> "Click Save! The transistor is now in your database. You'll see a confirmation message."

**[12:30 - Verify in search]**

> "Let's verify: go to Search Database, filter for SiC-MOSFET, and there it is! Click the name to see all the data you just entered. Beautiful!"

**[13:00 - Virtual datasheet]**

> "Pro tip: use the 'Preview on Virtual Datasheet' button before saving. This generates a PDF with all your curves plotted. Great way to spot errors before committing."

**[13:30 - Backup tip]**

> "Another pro tip: after adding transistors, export your database to JSON. This is your backup. Store it in Git or cloud storage. If something goes wrong, you can restore from the backup."

**[14:00 - Closing]**

> "That's how you add your own transistors! Yes, it takes time - maybe 15-20 minutes per device - but you only do it once, and then you have perfect data for all your future projects. In the next video, we'll export this transistor to PLECS. Thanks for watching!"

**[14:30 - End card]**

---

## Video 5: Exporting to Simulation Tools (12 minutes)

### Script

**[00:00 - Opening]**

> "You've got your transistor data in the database. Now let's get it into your simulation tool! Today we'll export to PLECS, MATLAB, and Simulink. By the end of this video, you'll be running thermal simulations with accurate models."

**[00:15 - Navigate to Export Tools]**

> "Switch to the Exporting Tools tab. Select a transistor from the dropdown - I'm using CREE C3M0060065J."

**[00:30 - Export format overview]**

> "You have six export formats: JSON for data interchange, MATLAB for scripts, PLECS for thermal models, Simulink for Simscape, GeckoCIRCUITS, and Virtual Datasheet for a nice PDF summary."

**[01:00 - PLECS export walkthrough]**

> "Let's start with PLECS, which is super popular for power electronics. Click the PLECS button. A dialog appears asking if you want to include thermal models. Say yes. Click Export."

**[01:30 - Show exported file]**

> "The file downloads as an XML file. Let's open it in a text editor. See all that XML? This is PLECS thermal library format. It contains: electrical parameters like on-resistance and voltage rating, thermal parameters like junction-to-case resistance, and switching loss tables."

**[02:15 - Import into PLECS]**

> "Now let's import it into PLECS. Open PLECS, go to PLECS → Thermal Library. Click Import, browse to our XML file, and open. Done! The transistor is now in your PLECS thermal library."

**[02:45 - Use in PLECS circuit]**

> "To use it: drag a MOSFET from the library into your circuit. Right-click, select Component Properties, then Thermal Description. You'll see our transistor in the list! Select it, and PLECS now uses accurate thermal and switching loss models."

**[03:30 - Run PLECS simulation demo]**

> "Let's run a quick simulation. I've set up a simple Buck converter at 400V input, 12V output, 100 kHz. Run the simulation... and look at that! Junction temperature rises to 85°C. With thermal models, we can predict reliability."

**[04:15 - MATLAB export]**

> "Back to the Transistor Database. Let's export to MATLAB. Select the transistor, click MATLAB, and export. This generates a .m script file."

**[04:45 - Show MATLAB script]**

> "Open the script in a text editor. It's pure MATLAB code! All the transistor parameters are defined as variables: v_abs_max equals 650, i_abs_max equals 120, r_ds_on equals 0.006. The switching loss data is in matrices. You can load this script and use the variables in your own MATLAB calculations."

**[05:30 - Load in MATLAB]**

> "In MATLAB: run the script. Now type 'v_abs_max' in the command window... 650! All variables are in your workspace. Super easy to use in custom scripts."

**[06:00 - Simulink export]**

> "Simulink export is similar but creates a .mat file. This is a binary format that loads directly into Simulink/Simscape models."

**[06:30 - Import into Simulink]**

> "In Simulink: use a Simscape MOSFET block, open parameters, and load the .mat file. The block now uses your custom transistor data. Perfect for detailed power loss and thermal analysis."

**[07:00 - GeckoCIRCUITS export]**

> "GeckoCIRCUITS users: export as .ipes format. This is GeckoCIRCUITS' native format. Open GeckoCIRCUITS, go to File → Import, select your .ipes file, and the transistor appears in your component library."

**[07:45 - Virtual Datasheet]**

> "The Virtual Datasheet export is special. It creates a PDF with all the curves and specifications, formatted like a real datasheet. This is perfect for documentation, design reviews, or sharing with colleagues who don't have the database installed."

**[08:30 - Show virtual datasheet]**

> "Let's generate one. Click Virtual Datasheet, export, and open the PDF. Beautiful! Page 1 has electrical ratings and thermal properties. Page 2 shows channel characteristics at different temperatures. Page 3 has switching loss curves. Page 4 shows capacitance curves. It's publication-quality!"

**[09:15 - JSON export use case]**

> "Finally, JSON export. This is the raw data format. Use it for: backing up your database, sharing transistors with colleagues, or importing into custom tools. It's human-readable and easy to parse programmatically."

**[09:45 - Bulk export tip]**

> "Pro tip: you can export multiple transistors at once. Go to Search Database, apply filters, and click 'Export Filtered Results'. All transistors in your search export as a ZIP file. Great for archiving or migrating databases."

**[10:15 - Re-import tip]**

> "Another tip: if you receive a JSON file from a colleague, you can import it. Go to Create Transistor, click 'Import from JSON', and load the file. The form auto-fills with all the data."

**[10:45 - Common export errors]**

> "Common issues: if export fails, check that the transistor has all required data. Missing switching loss curves? The PLECS export will fail. Check the browser console for error messages."

**[11:15 - Closing]**

> "That's everything about exporting! You now know how to get your transistor data into any simulation tool. Next video: using the Topology Calculator to design a complete converter. Don't miss it!"

**[11:30 - End card]**

---

## Video 6: Topology Calculator - Buck Converter Design (18 minutes)

### Script

**[00:00 - Opening]**

> "Today we're designing a complete Buck converter using the Topology Calculator. We'll size components, predict efficiency, and verify thermal performance - all without breadboarding a single circuit. Let's get started!"

**[00:20 - Design specs]**

> "Here's our design: input voltage 400V, output voltage 12V, output current 50A, switching frequency 100 kHz. This is a typical DC-DC converter for telecom or server power supplies."

**[00:45 - Navigate to Topology Calculator]**

> "Go to the Topology Calculator tab. First, select the topology. We want Buck - that's a step-down converter. The dropdown shows Buck, Boost, and Buck-Boost."

**[01:15 - Select transistor]**

> "Next, select the transistor. I'm using the CREE C3M0060065J we looked at earlier. 650V rating handles the 400V input with margin. 60A continuous rating handles our 50A output."

**[01:45 - Enter specifications]**

> "Now enter the design specs. Input voltage: 400. Output voltage: 12. Output current: 50. Switching frequency: 100000 - that's 100 kHz in Hertz."

**[02:15 - Inductance calculation]**

> "We need to specify inductance. How do we calculate it? For CCM operation, use the formula L = V_out × (1 - D) / (f_sw × delta_I), where delta_I is the current ripple. Let's target 20% ripple, so delta_I = 0.2 × 50A = 10A. Plugging in: L = 12 × (1 - 0.03) / (100000 × 10) = 11.6 microhenries. Let's use 15 microhenries to be safe."

**[03:15 - Output capacitance]**

> "Output capacitance: this filters the current ripple. For a target output voltage ripple of 50mV with 10A current ripple, C = I_ripple / (8 × f_sw × V_ripple) = 10 / (8 × 100000 × 0.05) = 25 microfarads. Use 50 microfarads for margin."

**[03:45 - Gate resistance]**

> "Gate resistance: this controls switching speed. Start with 10 ohms - a middle-of-the-road value. We'll optimize it in a minute using the slider."

**[04:00 - Click calculate]**

> "Click Calculate! The tool crunches the numbers... and here are our results!"

**[04:15 - Duty cycle result]**

> "First: duty cycle. The calculator shows 3%. That makes sense: D = V_out / V_in = 12 / 400 = 0.03 or 3%. The transistor is on for 3% of each switching period and off for 97%."

**[04:45 - Power loss breakdown]**

> "Power losses: conduction loss in the transistor is 1.2 watts. Switching loss is 3.5 watts. Diode loss is 0.8 watts. Total loss: 5.5 watts out of 600 watts output power."

**[05:15 - Efficiency result]**

> "Efficiency: 99.1%! That's excellent. SiC really shines here. With a silicon MOSFET, we'd be lucky to get 97% at this frequency."

**[05:45 - Thermal results]**

> "Junction temperature: 67°C with 25°C ambient. That's well below the 175°C max. We have plenty of thermal margin. The calculator assumes a small heatsink - check the thermal resistance parameters to see the assumptions."

**[06:15 - Waveforms]**

> "Scroll down to see waveforms. Here's the inductor current: triangular waveform with 10A peak-to-peak ripple, just as we designed. Gate voltage waveform shows clean switching with 10 ohm gate resistance."

**[06:45 - Gate resistance optimization]**

> "Now let's optimize! See the gate resistance slider at the bottom? Let's sweep it. Drag left to 5 ohms... efficiency increases to 99.3%! Lower gate resistance means faster switching and lower switching loss."

**[07:30 - But watch the tradeoffs]**

> "But look at the current waveform: more ringing now. That's the tradeoff. Faster switching causes more ringing due to parasitic inductance. In a real circuit, this could cause EMI issues. Let's try the other direction: increase to 20 ohms."

**[08:00 - Higher gate resistance]**

> "At 20 ohms: efficiency drops to 98.8%. Switching is slower, more loss, but cleaner waveforms. The optimal gate resistance depends on your layout and EMI requirements."

**[08:30 - Component stress]**

> "Check the component stress section. Peak transistor current: 55A - that's within the 60A rating. Peak voltage: 400V - within the 650V rating. RMS current: 3.5A - important for selecting the inductor."

**[09:00 - Losses at different loads]**

> "Here's a cool feature: click 'Scan Load Current'. The calculator sweeps output current from 10% to 100% and plots efficiency versus load. See how efficiency peaks at 75% load? That's typical for switching converters."

**[09:45 - Frequency sweep]**

> "You can also sweep frequency. Click 'Scan Frequency'. This plots efficiency versus switching frequency. See how efficiency drops at higher frequencies due to increased switching loss? But component sizes shrink. Classic tradeoff."

**[10:30 - Thermal analysis]**

> "Let's dive into thermal analysis. Click 'Show Thermal Model'. This displays the thermal equivalent circuit: junction to case, case to heatsink, heatsink to ambient. Each block shows resistance and temperature drop."

**[11:00 - Heatsink sizing]**

> "Say junction temperature is too high. How do we fix it? Add a better heatsink. Change R_th_heatsink_ambient from 10 K/W to 5 K/W. Recalculate... temperature drops to 52°C. Perfect!"

**[11:30 - Forced cooling]**

> "Or add a fan. With forced cooling, you can use R_th_heatsink_ambient of 2 K/W. Now temperature is 38°C - plenty of margin even at high ambient temperatures."

**[12:00 - Export results]**

> "Once you're happy with the design, export the results. Click 'Export Results' and choose PDF. This generates a design report with all specifications, waveforms, and efficiency plots. Perfect for design documentation."

**[12:30 - Compare topologies]**

> "Want to compare with a different topology? Change to Boost and recalculate. Wait, Boost won't work here - it steps up voltage, not down. But you could compare Buck versus Buck-Boost to see the efficiency difference."

**[13:00 - Save design]**

> "The calculator doesn't automatically save your design. So write down your final values: 15 µH inductor, 50 µF capacitor, 10 ohm gate resistance, 100 kHz frequency. You'll need these for the PCB design."

**[13:30 - Practical tips]**

> "Practical tips: First, the calculator assumes ideal components. Real inductors have DC resistance and core loss. Deduct 0.5-1% from the calculated efficiency. Second, layout matters! Parasitic inductance in the switching loop causes ringing. Keep traces short and use good grounding."

**[14:15 - Validation]**

> "Third, validate in simulation. Export the transistor to PLECS, build the Buck converter, and verify the results. The calculator gives you a great starting point, but detailed simulation catches second-order effects."

**[14:45 - When to use Buck-Boost]**

> "When to use Buck-Boost instead of Buck? When input voltage varies widely. Buck only works when V_in > V_out. If V_in can drop below V_out, you need Buck-Boost. But efficiency is slightly lower due to the extra switching device."

**[15:30 - Frequency selection]**

> "How to choose switching frequency? Higher frequency means smaller components but lower efficiency. For converters under 1 kW, 100-200 kHz is typical. Above 1 kW, consider 50-100 kHz. SiC enables higher frequencies than silicon."

**[16:15 - Parallel transistors note]**

> "If one transistor isn't enough, parallel multiple devices. The calculator doesn't support this directly, but you can approximate: divide current by N devices and multiply R_ds_on by 1/N."

**[16:45 - Closing]**

> "That's the complete workflow: spec your design, enter parameters, calculate, optimize gate resistance, verify thermal performance, and export results. You now have everything to design efficient power converters with confidence!"

**[17:15 - Next steps]**

> "Next steps: breadboard your design! Use the transistor you modeled, the calculated component values, and test it. Then compare measured efficiency to the prediction. It should be within a few percent if your data is accurate."

**[17:45 - End card]**

> "Thanks for watching this deep dive! Check out the other videos in the series for more advanced topics. And don't forget to star the project on GitHub - it's open source! See you next time!"

**[18:00 - End]**

---

## Production Notes

### Equipment Needed

- **Screen recording software**: OBS Studio (free), Camtasia, ScreenFlow
- **Microphone**: USB condenser mic (minimum Blue Yeti quality)
- **Video editing**: DaVinci Resolve (free), Adobe Premiere, Final Cut Pro
- **Thumbnail creation**: Canva, Photoshop

### Screen Recording Settings

- **Resolution**: 1920×1080 (1080p)
- **Frame rate**: 30 fps (60 fps for smooth animations)
- **Bitrate**: 5-10 Mbps
- **Format**: MP4 (H.264 codec)

### Audio Settings

- **Format**: 48 kHz, 16-bit
- **Noise reduction**: Apply in post-production
- **Normalization**: -3 dB peak
- **Room treatment**: Record in quiet space with soft furnishings

### Browser Setup for Recording

```bash
# Start backend
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Start frontend
cd transistordatabase/gui_web && npm run dev

# Open in browser at 1920x1080 resolution
# Zoom to 100% (Ctrl+0)
# Clear browser cache before recording (Ctrl+Shift+Del)
# Close all other tabs
```

### Editing Checklist

- [ ] Remove long pauses and "um"s
- [ ] Add intro/outro graphics (5-7 seconds each)
- [ ] Add text overlays for key points
- [ ] Add zoom effects for small UI elements
- [ ] Add background music (low volume, 10-15%)
- [ ] Color grade for consistent brightness
- [ ] Export with YouTube presets

### YouTube Upload Settings

- **Title**: "Transistor Database Tutorial: [Topic] | Power Electronics"
- **Description**: Include links to docs, GitHub repo, timestamps
- **Tags**: power electronics, transistor database, MOSFET, SiC, PLECS, power converter
- **Thumbnail**: 1280×720, bright colors, text overlay with topic
- **Playlist**: Add to "Transistor Database Tutorials"
- **Cards**: Link to next video at 80% mark
- **End screen**: Subscribe button + 2 related videos

### Timestamps Template

```
0:00 Introduction
0:15 Overview
1:30 Main Topic Part 1
5:00 Main Topic Part 2
8:45 Advanced Tips
12:00 Common Mistakes
14:30 Summary
15:00 Next Steps
```

---

## Additional Video Ideas

### Advanced Topics

1. **Python API Usage** (12 min) - Scripting with transistordatabase
2. **Batch Processing** (10 min) - Analyzing multiple transistors programmatically
3. **Custom Export Formats** (8 min) - Creating your own export templates
4. **Database Management** (10 min) - Organizing large transistor libraries
5. **Integration with LTspice** (15 min) - Exporting and simulating

### Application-Specific

1. **PFC Design** (20 min) - Designing a bridgeless PFC with the calculator
2. **Automotive Applications** (15 min) - Selecting transistors for EV inverters
3. **Solar Inverter Design** (18 min) - String inverter optimization
4. **Server PSU Design** (20 min) - High-efficiency telecom supplies
5. **Motor Drive Design** (18 min) - Three-phase inverter analysis

### Community Content

1. **Contributing to the Database** (12 min) - How to submit your transistors
2. **Reporting Issues** (5 min) - GitHub issues workflow
3. **Feature Requests** (8 min) - Requesting new functionality

---

**Ready to record? Follow these scripts and you'll create professional, helpful tutorials for the Transistor Database community!** 🎥
