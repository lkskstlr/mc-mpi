CC=/usr/local/Cellar/gcc/8.2.0/bin/gcc-8
CFLAGS=-O3
LDFLAGS=-lm

GRAPHICS_CFLAGS=-DGRAPHICS_ON
GRAPHICS_LDFLAGS=

SRC= main.c \
	absorption.c \
	dist_interaction.c \
	dist_sortie_couche.c \
	enable_all_partics.c \
	init_directions.c \
	init_domaine.c \
	init_graines.c \
	init_poids.c \
	init_positions.c \
	init_sig.c \
	init_wa.c \
	interaction.c \
	output.c \
	random.c \
	sortie_couche.c

all:mc_seq mc_seq_graphics

mc_seq_graphics: $(SRC)
	$(CC) $(CFLAGS) $(GRAPHICS_CFLAGS) -o $@ $^ $(LDFLAGS) $(GRAPHICS_LDFLAGS)

mc_seq: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f mc_seq mc_seq_graphics WA.out WA.png
