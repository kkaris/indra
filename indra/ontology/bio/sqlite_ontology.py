"""This module implements an SQLite back end to the
INDRA BioOntology."""

import os
import json
import zlib
import sqlite3
import logging
import threading
from collections import defaultdict
from indra.ontology.ontology_graph import IndraOntology
from indra.ontology.bio.ontology import CACHE_DIR, BioOntology


logger = logging.getLogger(__name__)


DEFAULT_SQLITE_ONTOLOGY = os.path.join(CACHE_DIR, 'bio_ontology.db')


def _zip(text):
    return zlib.compress(text.encode('utf-8'))


def _unzip(blob):
    return zlib.decompress(blob).decode('utf-8')


class SqliteOntology(IndraOntology):
    def __init__(self, db_path=DEFAULT_SQLITE_ONTOLOGY):
        super().__init__()
        self.db_path = db_path
        build_sqlite_ontology(db_path)
        self._local = threading.local()
        self._initialized = True

    @property
    def cur(self):
        return self._get_cursor()

    def _get_cursor(self):
        """Return a thread-local SQLite cursor, creating a connection on first use."""
        # Use hasattr rather than checking for None because each thread has its own
        # local instance of threading.local, which may or may not have the conn
        # attribute set
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(self.db_path)
            self._local.conn = conn
            self._local.cur = conn.cursor()
        return self._local.cur

    def initialize(self):
        pass

    def isa_or_partof(self, ns1, id1, ns2, id2):
        # Ontological parents (isa/partof ancestors) of a node are stored in
        # child_lookup, keyed by that node (see get_parents/child_rel).
        q = """SELECT children FROM child_lookup
               WHERE parent_id=? AND parent_ns=?
               LIMIT 1;"""
        self.cur.execute(q, (id1, ns1))
        res = self.cur.fetchone()
        if res is None:
            return False
        return '%s:%s|isa_or_partof' % (ns2, id2) in _unzip(res[0]).split(',')

    def child_rel(self, ns, id, rel_types):
        q = """SELECT children FROM child_lookup
               WHERE parent_id=? AND parent_ns=?
               LIMIT 1;"""
        if rel_types and 'isa' in rel_types or 'partof' in rel_types:
            rel_types |= {'isa_or_partof'}
        self.cur.execute(q, (id, ns))
        res = self.cur.fetchone()
        if res is None:
            yield from []
        else:
            children = _unzip(res[0]).split(',')
            for child in children:
                curie, rel_type = child.split('|', 1)
                if rel_type in rel_types:
                    yield tuple(curie.split(':', 1))

    def get_parents(self, ns, id):
        # Note that for isa/partof ontological child/parent is the
        # opposite of the graph-based child/parent
        return list(self.child_rel(ns, id, {'isa_or_partof'}))

    def get_children(self, ns, id, ns_filter=None):
        # Note that for isa/partof ontological child/parent is the
        # opposite of the graph-based child/parent
        children = list(self.parent_rel(ns, id, {'isa_or_partof'}))
        if ns_filter:
            children = [(cns, cid) for cns, cid in children
                        if cns in ns_filter]
        return children

    def parent_rel(self, ns, id, rel_types):
        q = """SELECT parents FROM parent_lookup
               WHERE child_id=? AND child_ns=?
               LIMIT 1;"""
        if rel_types and 'isa' in rel_types or 'partof' in rel_types:
            rel_types |= {'isa_or_partof'}
        self.cur.execute(q, (id, ns))
        res = self.cur.fetchone()
        if res is None:
            yield from []
        else:
            parents = _unzip(res[0]).split(',')
            for parent in parents:
                curie, rel_type = parent.split('|', 1)
                if rel_type in rel_types:
                    yield tuple(curie.split(':', 1))

    def get_node_property(self, ns, id, property):
        q = """SELECT properties FROM node_properties
               WHERE id=? AND ns=?
               LIMIT 1;"""
        self.cur.execute(q, (id, ns))
        res = self.cur.fetchone()
        if res is None:
            return None
        props = json.loads(res[0])
        return props.get(property)

    def get_name(self, ns, id):
        q = """SELECT name FROM node_properties
               WHERE id=? AND ns=?
               LIMIT 1;"""
        self.cur.execute(q, (id, ns))
        res = self.cur.fetchone()
        return res[0] if res is not None else None

    def get_id_from_name(self, ns, name):
        q = """SELECT id, properties FROM node_properties
               WHERE ns=? AND name=?;"""
        self.cur.execute(q, (ns, name))
        for node_id, props in self.cur.fetchall():
            if not json.loads(props).get('obsolete', False):
                return (ns, node_id)
        return None

    def nodes_from_suffix(self, suffix):
        self.cur.execute("SELECT ns, id FROM node_properties;")
        return [self.label(ns, id) for ns, id in self.cur.fetchall()
                if self.label(ns, id).endswith(suffix)]


def build_sqlite_ontology(db_path=DEFAULT_SQLITE_ONTOLOGY, force=False):
    # If the database already exists and we are not forcing a rebuild, return
    if os.path.exists(db_path) and not force:
        return

    if force:
        try:
            logger.info('Removing existing SQLite ontology at %s' % db_path)
            os.remove(db_path)
        except FileNotFoundError:
            pass

    # Initialize the bio ontology and build the transitive closure
    bio_ontology = BioOntology()
    bio_ontology.initialize()
    bio_ontology._build_transitive_closure()

    # Set up connection
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    logger.info('Building SQLite ontology at %s' % db_path)
    # Build the child/parent lookup contents in chunks
    chunk_size = 10000
    # Note: the transitive closure consists of pairs with the first element
    # being the ontological child and the second the parent. However,
    # in a graph representation isa/partof edges point from the ontological
    # child to the ontological parent. Here, we need to follow the graph-based
    # parent->child relationships, not the ontological ones.
    tc = sorted(bio_ontology.transitive_closure)
    all_children = defaultdict(set)
    all_parents = defaultdict(set)
    for i in range(0, len(tc), chunk_size):
        chunk = tc[i:i+chunk_size]
        chunk_values = [(parent.split(':', 1)[1], parent.split(':')[0],
                         child.split(':', 1)[1], child.split(':')[0])
                        for child, parent in chunk]
        for cid, cns, pid, pns in chunk_values:
            all_children[(pid, pns)].add(
                '%s:%s|%s' % (cns, cid, 'isa_or_partof'))
            all_parents[(cid, cns)].add(
                '%s:%s|%s' % (pns, pid, 'isa_or_partof'))

    for parent, child, data in bio_ontology.edges(data=True):
        parent_ns, parent_id = bio_ontology.get_ns_id(parent)
        child_ns, child_id = bio_ontology.get_ns_id(child)
        rel_type = data.get('type')
        if rel_type in {'isa', 'partof'}:
            continue
        all_children[(parent_id, parent_ns)].add(
            '%s:%s|%s' % (child_ns, child_id, rel_type))
        all_parents[(child_id, child_ns)].add(
            '%s:%s|%s' % (parent_ns, parent_id, rel_type))

    # Next, create child and parent lookup tables and populate them
    q = """CREATE TABLE child_lookup (
        parent_id TEXT NOT NULL,
        parent_ns TEXT NOT NULL,
        children BLOB NOT NULL,
        PRIMARY KEY (parent_id, parent_ns)
    ) WITHOUT ROWID;"""
    cur.execute(q)
    q = """CREATE TABLE parent_lookup (
        child_id TEXT NOT NULL,
        child_ns TEXT NOT NULL,
        parents BLOB NOT NULL,
        PRIMARY KEY (child_id, child_ns)
    ) WITHOUT ROWID;"""
    cur.execute(q)
    for (pid, pns), children in all_children.items():
        cur.execute("INSERT INTO child_lookup (parent_id, parent_ns, children) "
                    "VALUES (?, ?, ?);",
                    (pid, pns, _zip(','.join(children))))
    for (cid, cns), parents in all_parents.items():
        cur.execute("INSERT INTO parent_lookup (child_id, child_ns, parents) "
                    "VALUES (?, ?, ?);",
                    (cid, cns, _zip(','.join(parents))))

    # Create node property table
    # Here we just keep track of the namespace and ID,
    # and then put all the data into a json string
    q = """CREATE TABLE node_properties (
        id TEXT NOT NULL,
        ns TEXT NOT NULL,
        name TEXT,
        properties TEXT NOT NULL,
        PRIMARY KEY (id, ns)
    ) WITHOUT ROWID;"""
    cur.execute(q)

    for node in bio_ontology.nodes:
        ns, id = bio_ontology.get_ns_id(node)
        data = bio_ontology.nodes[node]
        name = data.get('name')
        props = {k: v for k, v in data.items() if k != 'name'}
        cur.execute("INSERT INTO node_properties (id, ns, name, properties) "
                    "VALUES (?, ?, ?, ?);", (id, ns, name, json.dumps(props)))

    cur.execute("CREATE INDEX idx_node_name ON node_properties (ns, name) "
                "WHERE name IS NOT NULL;")

    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    logger.info('Finished building SQLite ontology')
