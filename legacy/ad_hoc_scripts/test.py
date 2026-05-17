# Create the electron transfer diagram with simpler design without LaTeX
fig, ax = plt.subplots(figsize=(10, 8))

# Set background color
ax.set_facecolor('white')

# Draw the amide structure before the reaction
ax.text(0.1, 0.7, 'R-C(O)-NH-R\'', fontsize=14, ha='center', color='black')

# Arrow for NaH deprotonation
ax.annotate('', xy=(0.3, 0.7), xytext=(0.1, 0.7),
            arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

# Draw negative charge on nitrogen
ax.text(0.4, 0.7, 'N-', fontsize=12, ha='center', color='red')

# Draw MeI text and the methyl group reacting with nitrogen
ax.text(0.65, 0.75, 'MeI', fontsize=12, ha='center', color='purple')
ax.annotate('', xy=(0.8, 0.75), xytext=(0.65, 0.75),
            arrowprops=dict(arrowstyle="->", color='purple', lw=1.5))

# Draw the final product (tertiary amide)
ax.text(1.0, 0.7, 'R-C(O)-N(CH3)-R\'', fontsize=14, ha='center', color='black')

# Add indication for electron movement
ax.text(0.25, 0.8, 'Electron Movement', fontsize=12, ha='center', color='green')
ax.annotate('', xy=(0.25, 0.75), xytext=(0.1, 0.7),
            arrowprops=dict(arrowstyle="->", color='green', lw=1.5))

# Set plot limits and remove axis
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.5, 1.0)
ax.axis('off')

# Display the plot
plt.show()
